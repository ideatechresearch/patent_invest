from flask import Flask, jsonify, request
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
import jwt as PyJWT
from functools import wraps
import os
import uuid
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# JWT configuration
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key')  # 在生产环境中应该使用环境变量
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = 24  # hours

# Admin API Key configuration
ADMIN_API_KEY = os.getenv('ADMIN_API_KEY', '***')  # 在生产环境中应该使用环境变量

# API Key configuration
API_KEY = os.getenv('API_KEY', 'your-internal-api-key')  # 在生产环境中应该使用环境变量

# Database configuration
# DB_CONFIG = {
#     'host': 'localhost',
#     'port': 3307,
#     'database': 'hammer',
#     'user': 'root',  # Update this
#     'password': '123456'  # Update this
# }
DB_CONFIG = {
    'host': '10.10.10.5',
    'port': 3306,
    'database': '***',
    'user': 'dooven',  # Update this
    'password': '***'  # Update this
}

def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def generate_token(user_id):
    """Generate a JWT token for the user"""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION)
    }
    return PyJWT.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def init_api_keys_table():
    """Initialize the api_keys table if it doesn't exist"""
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_name VARCHAR(255) NOT NULL,
                api_key VARCHAR(255) NOT NULL UNIQUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_used_at DATETIME,
                is_active BOOLEAN DEFAULT TRUE,
                UNIQUE KEY unique_api_key (api_key)
            )
        """)
        conn.commit()
    except Error as e:
        print(f"Error creating api_keys table: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def log_api_key_usage(api_key, ip_address):
    """Log API key usage"""
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        # Update last_used_at in api_keys table
        cursor.execute("""
            UPDATE api_keys 
            SET last_used_at = CURRENT_TIMESTAMP 
            WHERE api_key = %s
        """, (api_key,))
        
        # Log the usage
        cursor.execute("""
            INSERT INTO api_key_logs (api_key, ip_address, user_agent, endpoint)
            VALUES (%s, %s, %s, %s)
        """, (api_key, ip_address, request.user_agent.string, request.endpoint))
        
        conn.commit()
    except Error as e:
        print(f"Error logging API key usage: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def require_admin_api_key(f):
    """Decorator to protect admin routes with admin API Key authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != ADMIN_API_KEY:
            return jsonify({'error': 'Admin API Key required'}), 403
        return f(*args, **kwargs)
    return decorated

def require_api_key(f):
    """Decorator to protect routes with API Key authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        logger.info(f"Received API Key request from IP: {request.remote_addr}")
        
        if not api_key:
            logger.warning(f"API Key missing from request from IP: {request.remote_addr}")
            return jsonify({'error': 'API Key is missing'}), 401
        
        # 如果是管理员 API Key，直接通过
        if api_key == ADMIN_API_KEY:
            logger.info(f"Admin API Key used from IP: {request.remote_addr}")
            request.user_name = 'admin'
            return f(*args, **kwargs)
        
        conn = get_db_connection()
        if not conn:
            logger.error("Database connection failed during API Key validation")
            return jsonify({'error': 'Database connection failed'}), 500
        
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT user_name, is_active 
                FROM api_keys 
                WHERE api_key = %s
            """, (api_key,))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"Invalid API Key used from IP: {request.remote_addr}")
                return jsonify({'error': 'Invalid API Key'}), 401
            
            if not result['is_active']:
                logger.warning(f"Inactive API Key used from IP: {request.remote_addr}")
                return jsonify({'error': 'API Key is inactive'}), 401
            
            logger.info(f"Valid API Key used by user: {result['user_name']} from IP: {request.remote_addr}")
            # Add user_name to request context for later use
            request.user_name = result['user_name']
            
            # Log the API key usage
            log_api_key_usage(api_key, request.remote_addr)
            
        except Error as e:
            logger.error(f"Database error during API Key validation: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()
        
        return f(*args, **kwargs)
    return decorated

@app.route('/api/auth/token', methods=['POST'])
@require_api_key
def get_token():
    """Generate a token for internal use"""
    try:
        logger.info(f"Generating token for user: {request.user_name}")
        token = generate_token(request.user_name)
        logger.info(f"Token generated successfully for user: {request.user_name}")
        return jsonify({'token': token})
    except Exception as e:
        logger.error(f"Error generating token for user {request.user_name}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/api-keys', methods=['POST'])
@require_admin_api_key
def create_api_key():
    """Create a new API key for a user"""
    data = request.get_json()
    if not data or 'user_name' not in data:
        return jsonify({'error': 'user_name is required'}), 400
    
    user_name = data['user_name']
    api_key = str(uuid.uuid4())
    
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO api_keys (user_name, api_key)
            VALUES (%s, %s)
        """, (user_name, api_key))
        conn.commit()
        
        return jsonify({
            'user_name': user_name,
            'api_key': api_key,
            'created_at': datetime.now().isoformat()
        })
    except Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/api/admin/api-keys', methods=['GET'])
@require_admin_api_key
def list_api_keys():
    """List all API keys"""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT user_name, api_key, created_at, last_used_at, is_active
            FROM api_keys
            ORDER BY created_at DESC
        """)
        results = cursor.fetchall()
        
        # Convert datetime objects to strings
        for result in results:
            if result['created_at']:
                result['created_at'] = result['created_at'].isoformat()
            if result['last_used_at']:
                result['last_used_at'] = result['last_used_at'].isoformat()
        
        return jsonify(results)
    except Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/api/admin/api-keys/<api_key>', methods=['DELETE'])
@require_admin_api_key
def revoke_api_key(api_key):
    """Revoke an API key"""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE api_keys 
            SET is_active = FALSE 
            WHERE api_key = %s
        """, (api_key,))
        conn.commit()
        
        if cursor.rowcount == 0:
            return jsonify({'error': 'API key not found'}), 404
        
        return jsonify({'message': 'API key revoked successfully'})
    except Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/api/admin/api-keys/logs', methods=['GET'])
@require_admin_api_key
def get_api_key_logs():
    """Get API key usage logs"""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT l.*, k.user_name
            FROM api_key_logs l
            JOIN api_keys k ON l.api_key = k.api_key
            ORDER BY l.created_at DESC
            LIMIT 1000
        """)
        results = cursor.fetchall()
        
        # Convert datetime objects to strings
        for result in results:
            if result['created_at']:
                result['created_at'] = result['created_at'].isoformat()
        
        return jsonify(results)
    except Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# Initialize the database tables when the application starts
init_api_keys_table()

@app.route('/api/task/<batch_no>/status', methods=['GET'])
@require_api_key
def get_task_status(batch_no):
    """Get the status of a task batch"""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT id, batch_no, title, status, created_at, completed_at
            FROM task_batch
            WHERE batch_no = %s
        """
        cursor.execute(query, (batch_no,))
        result = cursor.fetchone()
        
        if not result:
            return jsonify({'error': 'Task batch not found'}), 404
            
        return jsonify(result)
    except Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/api/task/<batch_no>/questions', methods=['GET'])
@require_api_key
def get_task_questions(batch_no):
    """Get all questions and their status for a task batch"""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT id, batch_no, question_no, origin_question, 
                   formatted_prompt, status, interface_result, 
                   result, created_at, updated_at
            FROM task_question
            WHERE batch_no = %s
            ORDER BY question_no
        """
        cursor.execute(query, (batch_no,))
        results = cursor.fetchall()
        
        return jsonify(results)
    except Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/api/task/<batch_no>/summary', methods=['GET'])
@require_api_key
def get_task_summary(batch_no):
    """Get the summary question and answer for a task batch"""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT id, batch_no, summary_question, summary_answer,
                   model, status, created_at, updated_at
            FROM task_summary_question
            WHERE batch_no = %s
        """
        cursor.execute(query, (batch_no,))
        result = cursor.fetchone()
        
        if not result:
            return jsonify({'error': 'Summary not found'}), 404
            
        return jsonify(result)
    except Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == '__main__':
    try:
        app.run(debug=True, port=5000, use_reloader=False)
    finally:
        # 确保所有数据库连接都被关闭
        if 'conn' in locals() and conn.is_connected():
            conn.close()
