import os
import base64
import requests, json

basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    # "sqlite:///project.db"
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://user:7777@10.10.10.5:3306/kettle?charset=utf8'
    SQLALCHEMY_COMMIT_ON_TEARDOWN = False
    SQLALCHEMY_TACK_MODIFICATIONS = True
    SQLALCHEMY_ECHO = True
    SECRET_KEY = '7777'
    DATA_FOLDER = 'data'
    NEO4J_HOST = '10.10.10.5'
    NEO4J_USERNAME = "***"
    NEO4J_PASSWORD = "7777"
    AIGC_HOST = 'aigc'
    AIGC_Service_Key = 'token-'
    QDRANT_HOST = '10.10.10.5'  # 'qdrant'
    QDRANT_URL = "http://10.10.10.5:6333"

    BAIDU_qianfan_API_Key = '***'
    BAIDU_qianfan_Secret_Key = '***'

    DashScope_Service_Key = '***' 
    Moonshot_Service_Key = "***"
    Silicon_Service_Key = '***'
    GLM_Service_Key = "***"

    def load_from_file(self, filename='config.ini'):
        import configparser
        # 从文件加载配置
        # 这里可以使用类似ConfigParser的方法来加载INI文件等
        # 从INI文件加载配置
        config_parser = configparser.ConfigParser()
        config_parser.read(filename)
        # 读取数据库配置
        # self.host = config_parser.get('Database', 'host', fallback=self.host)


def get_baidu_access_token(API_Key=Config.BAIDU_API_Key, Secret_Key=Config.BAIDU_Secret_Key):
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={API_Key}&client_secret={Secret_Key}"

    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")


def xor_encrypt_decrypt(data, key):
    return ''.join(chr(ord(c) ^ ord(k)) for c, k in zip(data, key * (len(data) // len(key) + 1)))


def encode_id(raw_id):
    return base64.urlsafe_b64encode(raw_id.encode()).decode().rstrip('=')


def decode_id(encoded_id):
    padded_encoded_id = encoded_id + '=' * (-len(encoded_id) % 4)
    return base64.urlsafe_b64decode(padded_encoded_id.encode()).decode()


def decode_id(encoded_id):
    return encoded_id

if __name__ == "__main__":
    key = Config.SECRET_KEY
    original_data = '123'

    # 加密数据
    encrypted_data = xor_encrypt_decrypt(original_data, key)
    encoded_id = encode_id(encrypted_data)
    print("加密后的数据:", encrypted_data, encoded_id)

    # 解密数据
    decoded_id = decode_id(encoded_id)
    decrypted_data = xor_encrypt_decrypt(decoded_id, key)  # encrypted_data
    print("解密后的数据:", decrypted_data, decoded_id)
