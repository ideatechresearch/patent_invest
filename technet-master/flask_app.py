# -*- coding: utf-8 -*-
from flask import session, request, redirect, url_for, render_template, render_template_string
from flask import jsonify, send_file, g, send_from_directory, Response
from flask import Flask
from database import *
from select_stop_words import *
from qdrant_net import QdrantClient, Graph, VDBRelationships, most_similar_embeddings, field_match, empty_match, \
    qdrant_livez
import plotly.express as px
import plotly.io as pio
import pandas as pd
import string, time, os, re
import inspect
from lda_topics import LdaTopics
import logging


class IgnoreHeadRequestsFilter(logging.Filter):
    def filter(self, record):
        return "HEAD / HTTP/1.0" not in record.getMessage()


lt = LdaTopics()
swg = StopWordsFlag()
vdr = VDBRelationships()

DEBUG_MODE = False  # .getenv('PYCHARM_HOSTED', '0') == '1'  # -e PYCHARM_HOSTED=0
if DEBUG_MODE:
    from config_debug import *  # Config
else:
    from config import *  # 配置信息

app = Flask(__name__, template_folder='templates')
app.secret_key = Config.SECRET_KEY  # 设置表单交互密钥
app.config['DATA_FOLDER'] = Config.DATA_FOLDER  # data
app.config["SQLALCHEMY_DATABASE_URI"] = Config.SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = Config.SQLALCHEMY_COMMIT_ON_TEARDOWN  # false
app.config['SQLALCHEMY_POOL_RECYCLE'] = 3600  # 适用于SQLAlchemy的连接池,设置连接重用时间为1小时

client = QdrantClient(url=Config.QDRANT_URL)
Baidu_Access_Token = get_baidu_access_token()


# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True #动态追踪修改设置

def get_user_id():
    username = session.get("username")
    if username:
        return swg.get_user(username)[0]
    return -1


# def get_db():
#     """Opens a new database connection if there is none yet for the
#     current application context.
#     """
#     if not hasattr(g, 'db'):#'db' not in g
#         g.db = connect_db()
#     return g.db

# @app.teardown_appcontext
# def teardown_db(error):
#     """Closes the database again at the end of the request."""
#     g_db = g.pop('db', None)
#     if g_db is not None:  # hasattr(g, 'db')
#         g_db.close()

# @app.before_request
# def block_head_requests():
#     if request.method == 'HEAD' and request.path == '/':
#         return "HEAD requests are not allowed", 405

@app.route('/search', methods=['GET', 'POST'])
def search():
    names, data, detail = [], [], []
    if request.method == 'POST':
        action = request.form.get('action')  # get('event_type')
        if action == 'search_words':
            txt = request.form['search_words'].strip()
            detail = swg.search_words(txt)
            if detail.shape[0] > 0:
                detail = words_detail_links(detail, '/details')
                return render_template('search.html', table=detail.to_html(escape=False, classes='custom-table'))
            return render_template('search.html', inform=f'未找到相关词：{txt}')

        if action == 'search_patent':
            txt = request.form['search_patent'].strip()
            empty_abstract = empty_match(field_key='摘要长度')
            if request.form.get('abstract5', '0') == '0':
                match = empty_abstract
                not_match = []
            else:
                not_match = empty_abstract
                match = []

            data = most_similar_embeddings(txt, '专利_bge_189', client,
                                           topn=int(request.form.get('topn', 10)),
                                           match=match, not_match=not_match,
                                           score_threshold=float(request.form.get('score_threshold', 0.0)),
                                           access_token=Baidu_Access_Token)

            detail = patents_detail_links(data, base_url='/details')
            if len(detail):
                return render_template('search.html', inform=f'{txt}',
                                       table=detail.to_html(escape=False, classes='custom-table'))

        if action == 'search_topic':
            txt = request.form['search_topic'].strip()
            if lt.notload():
                lt.load('xjzz', len_below=2, top_n_topics=4, minimum_probability=0.03, weight_threshold_topics=0.03)
            if txt:
                vec = lt.encode(txt)
                search_hit = client.search(collection_name='专利_先进制造_w2v_lda_120',
                                           query_vector=vec,
                                           score_threshold=float(request.form.get('score_threshold', 0.0)),
                                           limit=int(request.form.get('topn', 10)))

                data = [(p.payload, p.score) for p in search_hit]
                detail = patents_detail_links(data, base_url='/details')
            if len(detail):
                return render_template('search.html', inform=f'{txt}',
                                       table=detail.to_html(escape=False, classes='custom-table'))

        if action == 'search_co':
            txt = request.form['search_co'].strip()
            match = field_match(field_key='行业', match_values=request.form.get('hy', 'all'))
            empty_abstract = empty_match(field_key='简介长度')
            not_match = []
            if request.form.get('abstract4', '0') == '0':
                match += empty_abstract
            else:
                not_match = empty_abstract

            data = most_similar_embeddings(txt, '行业公司名简介_25k', client,
                                           topn=int(request.form.get('topn', 10)),
                                           score_threshold=float(request.form.get('score_threshold', 0.0)),
                                           match=match, not_match=not_match,
                                           access_token=Baidu_Access_Token)

            detail = companys_detail_links(data, base_url='/details')
            if len(detail):
                return render_template('search.html', inform=f'{txt}',
                                       table=detail.to_html(escape=False, classes='custom-table'))

        if action == 'search_sim_words':
            txt = request.form['search_sim_words'].strip()
            tokens = re.split(r'[^\w\s]| ', txt)

            names, data = vdr.similar_by_names(tokens, vdb_key='Word_' + request.form.get('hy', 'all'),
                                               topn=int(request.form.get('topn', 10)),
                                               exclude=swg.stop_words if request.form.get('exclude') == '1' else [],
                                               duplicate=int(request.form.get('duplicate', 0)),
                                               score_threshold=float(request.form.get('score_threshold', 0.0)))

        if action == 'search_sim_co':
            txt = request.form['search_sim_co'].strip(string.punctuation)
            tokens = re.split(r'[^\w\s]| ', txt)
            names, data = vdr.similar_by_names(tokens, vdb_key='Co_' + request.form.get('hy', 'all'),
                                               topn=int(request.form.get('topn', 10)),
                                               duplicate=int(request.form.get('duplicate', 0)),
                                               score_threshold=float(request.form.get('score_threshold', 0.0)))

        if data:
            detail = pd.concat([pd.Series(next, name=w).explode() for w, next in data], axis=1)
            if detail.shape[0] >= 3 and detail.shape[0] <= 24:
                return render_template('search.html', inform=f'{names}',
                                       table=detail.to_html(classes='custom-table'))

        return render_template('search.html', inform=f'{names}', data=data)

    return render_template('search.html')


@app.route('/relationship', methods=['GET', 'POST'])
def relationship():
    if request.method == 'POST':
        action = request.form.get('action')  # get('event_type')
        if action == 'search_relationships':
            txt = request.form['search_relationships'].strip().lower()
            data = vdr.match_relationships(txt)
            return render_template('relationship.html', inform=txt, data=data)

        uid, inf = swg.get_user(session.get("username"))
        if action == 'neo4j':
            return redirect(f'http://{Config.NEO4J_HOST}:7474/browser/' if uid >= 0 else 'https://neo4j.com/')
        if action == 'create_relationships':
            txt = request.form['create_relationships'].strip(string.punctuation).strip().lower()
            if uid < 0:
                return render_template('relationship.html', inform=f'{txt}:用户未注册或登录！')

            params = relationship_params(request.form, uid, 'Word_')
            names_depth, relationships_edges = vdr.SimilarRelationships(txt, create=1, **params)

            return render_template('relationship.html', inform=f'Word_{txt}:{names_depth}', data=relationships_edges)

        if action == 'create_relationships_co':
            txt = request.form['create_relationships_co'].strip(string.punctuation).strip().lower()
            if uid < 0:
                return render_template('relationship.html', inform=f'{txt}:用户未注册或登录！')

            params = relationship_params(request.form, uid, 'Co_')
            names_depth, relationships_edges = vdr.SimilarRelationships(txt, create=1, **params)

            return render_template('relationship.html', inform=f'Co_{txt}:{names_depth}', data=relationships_edges)

    return render_template('relationship.html')


def relationship_params(args_form, uid, key_prefix=''):
    nums = re.split(r'[^\w\s]| ', args_form.get('layers', '').strip())
    nums = [int(i) for i in nums if i.isdigit()]

    params = dict()
    params['vdb_key'] = key_prefix + args_form.get('hy', 'all')
    params['max_calc'] = 100 if uid < 0 else 300
    params['max_node'] = 1000 if uid < 0 else 3000
    params['layers'] = [i for i in nums if i >= 1 and i <= params['max_node']]
    params['batch'] = int(args_form.get('batch', 1))
    params['width'] = int(args_form.get('width', 3))
    params['max_depth'] = int(args_form.get('depth', 3))
    params['duplicate'] = int(args_form.get('duplicate', 3))
    params['score_threshold'] = float(args_form.get('score_threshold', 0.0))

    if key_prefix == 'Word_':
        params['key_radius'] = args_form.get('key_radius', '')
        params['exclude'] = swg.stop_words if args_form.get('exclude') == '1' else []

    # print(request.args.to_dict(), '\n', name, params)
    return params


@app.route('/show_relationships', methods=['GET', 'POST'])
def show_relationships():
    uid, inf = swg.get_user(session.get("username"))
    name = request.args.get('name', '').strip(string.punctuation).strip().lower()
    params = relationship_params(request.args, uid, key_prefix=request.args.get('key_prefix', ''))
    data = vdr.SimilarRelations(name, draw=int(request.args.get('draw', 0)), **params)

    # names_depth, relationships_edges = vdr.SimilarRelationships(name, create=0, **params)
    #
    # if not len(relationships_edges):
    #     nodes = [{"name": name, "id": 1, "depth": 0, "radius": 20}]
    #     return jsonify({"nodes": nodes, "edges": []})
    #
    # data = vdr.SimulationNodes(names_depth, relationships_edges, params['vdb_key'], params.get('key_radius', ''))

    return jsonify(data)


@app.route('/node_relations/<int:node_id>/<string:node_name>', methods=['GET', 'POST'])
def node_relations(node_id, node_name):
    request_data = request.get_json()
    existing_nodes = request_data.get('existingNodes', [])
    # print(f"Node name: {node_name}  Existing nodes: {existing_nodes} \n {names_depth}")

    key_prefix = request.args.get('key_prefix', '')
    params = {}
    params['vdb_key'] = key_prefix + request.args.get('hy', 'all')
    params['width'] = int(request.args.get('nodewidth', 1))
    params['duplicate'] = int(request.args.get('duplicate', 3))
    params['score_threshold'] = float(request.args.get('score_threshold', 0.0))
    if key_prefix == 'Word_':
        params['exclude'] = swg.stop_words if request.args.get('exclude') == '1' else []

    data = vdr.SimilarRelation(node_id, node_name, existing_nodes, **params)
    return jsonify(data)


@app.route('/similar/<string:id>/<string:name>', methods=['POST'])
def similar(id, name):
    decrypted_id = xor_encrypt_decrypt(decode_id(id), Config.SECRET_KEY)
    try:
        index = int(decrypted_id)
    except:
        index = -1
    data = vdr.similar(_id=index, name=name, not_ids=[], vdb_key='Word_all',
                       topn=10, exclude=[], score_threshold=0.0)
    return jsonify({"similar_next": data})


@app.route('/get_words', methods=['GET', 'POST'])
def get_words():
    uid, inf = swg.get_user(session.get("username"))
    inform = ''
    if request.method == 'POST':
        action = request.form.get('action')
        if uid >= 0:
            if action == 'set_flag':
                mask, flags = swg.set_stop_flag(request.form.getlist('item'), uid=uid)
                logs = [UserLog(user_id=uid, word=w, stoped=bool(f)) for w, f in zip(swg.word_data[mask].index, flags)]
                db.session.add_all(logs)
                table_records = swg.flag_table(mask=mask).to_dict(orient='records')  # 转换为 SQLAlchemy 对象
                for record in table_records:
                    db.session.merge(StopWords(**record))  # 使用 merge 合并来添加或更新对象
                db.session.commit()  # session.flush()
                inform = f"您已查阅：{inf.get('readn', 0)}，已标记：{inf.get('stopn', 0)} 停用词"
            if action == 'call_back':
                words, detail = swg.call_back_words(uid=uid)
                detail = words_detail_links(detail)
                detail = words_detail_absrtact(words, detail, absrtact=int(request.form.get('absrtact', 1)))

                return render_template('words.html', wordslist=words,
                                       table=detail.to_html(escape=False, classes='custom-table'))
        else:
            inform = '用户未注册或登录！'

    words, detail = swg.call_select_words(limit=int(request.form.get('limit', 4)),  # inf.get('limit', 4)
                                          randoms=int(request.form.get('randoms', 1)),
                                          cross=inf.get('cross', False), uid=uid)

    detail = words_detail_links(detail)
    detail = words_detail_absrtact(words, detail, absrtact=int(request.form.get('absrtact', 1)))

    html_detail = detail.to_html(escape=False, classes='custom-table')  # index=False

    return render_template('words.html', wordslist=words, table=html_detail, inform=inform)
    # ','.join(swg.user_data[0]['task_words']),','.join(swg.get_stop_words(uid=0))


def companys_detail_links(data, base_url='/details'):
    if not data:
        return []
    payloads, scores = zip(*data)  # [{}]
    if list(payloads):
        df = pd.DataFrame(payloads)
        df['score'] = list(scores)
        df['encode_id'] = df['公司序号'].map(
            lambda x: encode_id(xor_encrypt_decrypt(str(x), Config.SECRET_KEY)))
        mask = df['行业'] != '金融科技'
        df.loc[mask, '公司简称'] = df[mask].apply(
            lambda row: f'<a href="{base_url}/company/{row["encode_id"]}" target="_blank">{row["公司简称"]}</a>',
            axis=1)
        mask = df['行业'].isin(['医疗健康', '先进制造'])
        df.loc[mask, '工商全称'] = df[mask].apply(
            lambda row: f'<a href="{base_url}/patent_invest/{row["encode_id"]}" target="_blank">{row["工商全称"]}</a>',
            axis=1)

        return df[['公司简称', '工商全称', '行业', 'score']]
    return []


def patents_detail_links(data, base_url='/details'):
    if not data:
        return []
    payloads, scores = zip(*data)  # [{}]
    if list(payloads):
        df = pd.DataFrame(payloads).rename(
            columns={'标题 (中文)': 'patent_title', "公开（公告）号": 'publication_number'})
        df['score'] = list(scores)
        if '序号' in df.columns:
            df['encode_id'] = df['序号'].map(
                lambda x: encode_id(xor_encrypt_decrypt(str(x), Config.SECRET_KEY)))
            df['publication_number'] = df.apply(
                lambda
                    row: f'<a href="{base_url}/index/{row["encode_id"]}" target="_blank">{row["publication_number"]}</a>',
                axis=1)

        df['encode_id'] = df['申请号'].map(
            lambda x: encode_id(xor_encrypt_decrypt(x, Config.SECRET_KEY)))
        df['application_number'] = df.apply(
            lambda row: f'<a href="{base_url}/patent/{row["encode_id"]}" target="_blank">{row["申请号"]}</a>', axis=1)

        return df[['patent_title', 'Co', 'publication_number', 'application_number', 'score']]
    return []


def words_detail_links(table, base_url='/details'):
    df = table.copy()
    # 在DataFrame中添加链接列 target="_blank" View
    mask = df['w2v'] >= 0
    df.loc[mask, 'encode_id'] = df.loc[mask, 'w2v'].map(
        lambda x: encode_id(xor_encrypt_decrypt(str(x), Config.SECRET_KEY)))
    df.index = df.apply(
        lambda
            row: f'<span class="clickable" title="显示相似词" data-info={row["encode_id"]} style="cursor: pointer; user-select: none; color: #377ba8;"> {row.name}</span>' if
        row['w2v'] >= 0 else row.name, axis=1)
    df['encode_id'] = df['index'].map(
        lambda x: encode_id(xor_encrypt_decrypt(str(x), Config.SECRET_KEY)))
    df['publication_number'] = df.apply(
        lambda row: f'<a href="{base_url}/index/{row["encode_id"]}" target="_blank">{row["publication_number"]}</a>',
        axis=1)
    df['encode_id'] = df['application_number'].map(
        lambda x: encode_id(xor_encrypt_decrypt(str(x), Config.SECRET_KEY)))
    df['application_number'] = df.apply(
        lambda
            row: f'<a href="{base_url}/{row["table_name"]}/{row["encode_id"]}" target="_blank">{row["application_number"]}</a>',
        axis=1)  # '/details/patent',

    return df.drop(columns=['encode_id', 'table_name', 'w2v'])


def words_detail_absrtact(words, table, absrtact=1):
    df = table.copy()
    mask = df['index'].duplicated(keep='first')
    df.loc[mask, ['patent_title']] = '同上'

    def highlight(text, words):
        for word in words:
            text = text.replace(word, f'<span style="background-color: yellow; color: black;">{word}</span>')
        return text

    if absrtact:
        ids = tuple(df['index'].to_list()) if df.shape[0] > 1 else (df.iloc[0, 'index'],)
        query = text(f'SELECT 序号,`摘要 (中文)` FROM `融资公司专利-202406` WHERE 序号 in :ids')
        result = db.session.execute(query, {'ids': ids})
        result_dict = {row[0]: row[1] for row in result.fetchall()}  # result.fetchone()
        df['abstract'] = df['index'].map(result_dict)
        df.loc[mask, 'abstract'] = '同上'
        df.loc[~mask, 'abstract'] = df.loc[~mask, 'abstract'].apply(lambda x: highlight(x, words))

    return df.drop(columns=['index', '阅读标记', '停用标记'])


@app.route('/details/<string:to>/<string:id>')
def details(to, id):
    uid, inf = swg.get_user(session.get("username"))
    if uid < 0:
        return jsonify({'error': 'The user is not registered or logged in!'}), 400

    # 根据ID查询数据库
    decrypted_id = xor_encrypt_decrypt(decode_id(id), Config.SECRET_KEY)
    if to == 'index':
        table_name = '融资公司专利-202406'
        column_name = '序号'
    elif to == 'patent':
        table_name = '融资公司专利-202406'
        column_name = '申请号'  # '公开（公告）号'
        query = text('SELECT table_name FROM `融资公司专利-202406` WHERE 申请号 = :id')
        result = db.session.execute(query, {'id': decrypted_id})
        row = result.fetchone()  # [dict(row) for row in result.fetchall()]
        if row:
            table_name = row[0]
    elif to == 'company':
        table_name = '公司融资数据修正-20240301'
        column_name = '公司序号'
    elif to == 'patent_invest':
        table_name = 'patent_invest_2024_先进制造_医疗健康'
        column_name = '公司序号'
    elif to in ('patent_incopat_202101_202211', 'patent_incopat_202212_202312',
                'patent202210', 'patent202309', 'patent202404',):
        table_name = to
        column_name = '申请号'
    else:
        return jsonify({'error': 'Invalid type specified'}), 400

    detail_query = text(f'SELECT * FROM `{table_name}` WHERE `{column_name}` = :id limit 32')

    detail_df = pd.read_sql_query(detail_query, con=db.engine, params={'id': decrypted_id}).T

    detail_html = detail_df.replace('', np.nan).dropna(how='all').to_html()
    # 定义详细信息页面的HTML模板
    detail_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Detail View</title>
        <style>
        body {
            background-color: #333333; /* 深灰色背景 */
            color: #CCCCCC;           /* 浅灰色文字 */
            font-family: Arial, sans-serif;
        }
        </style>
    </head>
    <body>
        <h1>Detail Information</h1>
        <!-- 在此处嵌入详细信息表格 -->
        {{ detail_html|safe }}
    </body>
    </html>
    """

    # 渲染HTML模板并传递详细信息数据
    return render_template_string(detail_template, detail_html=detail_html)


@app.route('/chat')
def chat():
    return render_template('chat.html')


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()  # await
    question = data['question']
    # content_info = request.data.decode('utf-8')
    # content_json = json.loads(content_info)
    # import openai
    # response = openai.Completion.create(
    #     engine="davinci",  # Choose the engine (e.g., davinci, curie, etc.)
    #     prompt=question,
    #     max_tokens=150  # Adjust based on your needs
    # )
    #
    # answer = response.choices[0].text.strip()

    return jsonify({'answer': question})


def get_function_parameters(func):
    signature = inspect.signature(func)
    parameters = signature.parameters
    return [param for param in parameters]


@app.route('/plot', methods=['GET', 'POST'])
def plot():
    patent_types = ['发明授权', '发明申请', '实用新型', '外观设计']  # df['occupation'].unique()
    columns = ['权利要求数量', '独立权利要求数量', '文献页数']  # claim_number
    if request.method == 'POST':
        patent_type = request.form.get('patent_type', patent_types[0])
        col = request.form.get('columns', columns[0])
        query = f"SELECT `{col}` FROM patent202309 WHERE `专利类型` = '{patent_type}'"
        results = pd.read_sql(query, con=db.engine)
        # params = get_function_parameters(px.scatter)
        fig = px.histogram(results, x=col, title=f'Distribution for {patent_type}')
        graph_html = pio.to_html(fig, full_html=False)

        return render_template('plot.html', patent_type=patent_types, columns=columns, graph_html=graph_html)

    return render_template('plot.html', patent_type=patent_types, columns=columns)
    # text("SELECT 权利要求数量 FROM patent202309 WHERE `专利类型` = :patent_type")
    # with db.engine.connect() as connection:
    #     results = connection.execute(query.params(patent_type=patent_type)).fetchall()


@app.route('/get_user_info', methods=['GET', 'POST'])
def get_user_info():
    if request.method == 'POST':
        swg.reset_user_data()
        swg.get_stop_words(uid=-1)
    table = pd.DataFrame(swg.user_data)
    if not table.empty:
        table.sort_values('readn', inplace=True, ascending=False)  # table['readn'].sum()
    count = len(swg.stop_words)
    inf = ','.join(np.random.choice(swg.stop_words, size=min(300, count), replace=False)) \
        if count > 0 else ''
    return render_template('user_info.html',
                           table=table.to_html(classes='custom-table', index=False),
                           num=f'{count}/{swg.word_data.shape[0]}',
                           inform=inf)
    # return f'''
    #   <html>
    #       <body>
    #           <h1>用户信息表</h1>
    #           <div>{info.to_html()} </div>
    #       </body>
    #   </html>
    # '''


@app.route('/set_user', methods=['GET', 'POST'])
def set_user():
    username = session.get("username")
    uid, inf = swg.get_user(username)
    if uid < 0:
        return render_template('user.html', inform=f'用户未注册或登录！')
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'set_user':
            inf = swg.set_user(uid=uid,
                               cross=1 if request.form['cross'].lower() == "true" else 0)

            users = User.query.filter_by(user_id=uid)  # .first()
            users.update({User.username: inf['name'], User.cross: inf['cross']})
            db.session.commit()
            # db.session.close()

        if action == 'set_words':
            inf = {}
            mask, words = [], []
            if request.form['reset']:
                mask, words = swg.set_words_flag(request.form['reset'], uid, 0)
                inf['移除停用词'] = words
            elif request.form['doset']:
                mask, words = swg.set_words_flag(request.form['doset'], uid, 1)
                inf['增加停用词'] = words

            if sum(mask) > 0:
                logs = [UserLog(user_id=uid, word=w, stoped=1) for w in words]
                db.session.add_all(logs)
                table_records = swg.flag_table(mask=mask).to_dict(orient='records')  # 转换为 SQLAlchemy 对象
                for record in table_records:
                    words = StopWords.query.filter_by(word=record['word'])
                    if words:
                        words.update({StopWords.stop_flag: record['stop_flag']})
                    else:
                        db.session.merge(StopWords(**record))  # 使用 merge 合并来添加或更新对象
                db.session.commit()  # session.flush()
            else:
                inf['找不到词'] = words

        return render_template('user.html', username=username, inform=f'{inf}')

    stop_words = swg.get_stop_words(uid=uid)
    if stop_words.shape[0] > 0:
        return render_template('user.html', username=username,
                               inform=f'当前用户已标记停用词：{",".join(np.random.choice(stop_words, size=min(300, stop_words.shape[0]), replace=False))}')

    return render_template('user.html', username=username)

    # return f'''
    #   <html>
    #       <body>
    #        <h2>Selection Result</h2>
    #     <ul>
    #         {% for item, value in info.items() %}
    #             <li>{{ item }} - {{ "Selected" if selected else "Not selected" }}</li>
    #         {% endfor %}
    #     </ul>
    #       </body>
    #   </html>
    # '''


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        password = request.form['password']
        if password == app.secret_key:
            session['username'] = 'admin'
            return redirect('/admin/do')
        else:
            return "非管理员！<a href='/'>首页</a>"
    return '''
        <form method="post">
            <p><input type=password name=password>
            <p><input type=submit value=Login>
        </form>
        '''


host_index = 0


@app.route('/admin/do', methods=['GET', 'POST'])
def admin_do():
    checklist = ['测试用户', '重置清零', '导入标记', '保存标记', '记录停用词', '导入用户', '导入数据库', '备份数据库',
                 '切换向量库', '检测向量库', '连接图库']
    hosts = [Config.QDRANT_HOST, '47.110.156.41', '10.10.10.5', ]

    if request.method == 'POST':
        if session.get("username") != 'admin':
            return render_template("admin.html", checklist=checklist, inform='非管理员!')

        selected = request.form.getlist('item')
        ret = ''
        if '测试用户' in selected:
            swg.register_user('test')
        if '重置清零' in selected:
            swg.reset()
            session.clear()
        if '导入标记' in selected:
            swg.load_flag()
        if '保存标记' in selected:
            swg.save_flag()
        if '记录停用词' in selected:
            swg.save_data()
        if '导入用户' in selected:
            table = pd.read_sql(f'select * from users', con=db.engine).set_index('user_id')
            table = table[['username', 'cross']].rename(columns={'username': 'name'})
            table['readn'] = 0
            table['stopn'] = 0
            swg.user_data = table.to_dict(orient='records')
            ret += f"update users: {table.shape[0], swg.user_data}"
        if '导入数据库' in selected:
            table = pd.read_sql(f'select word,read_flag,stop_flag from stop_words', con=db.engine).set_index('word')
            if '阅读标记' not in swg.word_data.columns:
                swg.word_data['阅读标记'] = 0
                swg.word_data['停用标记'] = 0

            table.columns = ['阅读标记', '停用标记']
            swg.word_data.update(table)
            ret += f"update stop_words: {table.shape[0], swg.word_data.columns}"
        if '备份数据库' in selected:
            if len(swg.stop_words):
                table = swg.flag_table()
                try:
                    # 在应用上下文中执行数据库操作
                    with app.app_context():
                        table.to_sql('stop_words_backup', con=db.engine, if_exists='replace', index=False)
                except Exception as e:
                    # 捕获并处理可能的异常
                    ret += f"An error occurred: {str(e)}"
                    engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"])
                    conn = engine.connect()
                    table.to_sql('stop_words_backup', con=engine, if_exists='replace', index=False)
                    conn.close()
                finally:
                    db.session.commit()
                    ret += f"backup: {table.shape[0]}"

        if '切换向量库' in selected:
            global host_index
            global client
            host_index = (host_index + 1) % len(hosts)  # 循环切换 host 索引
            client = QdrantClient(host=hosts[host_index], grpc_port=6334, prefer_grpc=True)
            vdr.switch_clients(client)
            ret += f"clients change to: {hosts[host_index]}"
        if '检测向量库' in selected:
            global Baidu_Access_Token
            Baidu_Access_Token = get_baidu_access_token()

            res, status = qdrant_livez(hosts[host_index])
            ret += str(res)
        if '连接图库' in selected:
            try:
                graph = Graph(f"bolt://{Config.NEO4J_HOST}:7687", auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD))
                vdr.graph = graph
            except:
                ret += 'neo4j graph connection unavailable'

            # if action == 'set_words':
        #     if request.form['reset']:
        #         swg.set_words_flag(request.form['reset'],-1, 0)
        return render_template("admin.html", checklist=checklist, inform=f'{selected},{ret}')

    return render_template("admin.html", checklist=checklist)


@app.route('/download', methods=['GET', 'POST'])
def download():
    """  文件下载  """
    timelist = []  # 获取指定文件夹下文件并显示
    Foder_Name = []  # 文件夹下所有文件
    Files_Name = []  # 文件名
    file_dir = app.config['DATA_FOLDER']  # ./upload
    # 获取到指定文件夹下所有文件
    lists = os.listdir(file_dir + '/')

    # 遍历文件夹下所有文件
    for i in lists:
        # os.path.getatime => 获取对指定路径的最后访问时间
        timelist.append(time.ctime(os.path.getatime(file_dir + '/' + i)))

    # 遍历文件夹下的所有文件
    for k in range(len(lists)):
        # 单显示文件名
        Files_Name.append(lists[k])
        # 获取文件名以及时间信息
        Foder_Name.append(lists[k] + " -------------- " + timelist[k])

    return render_template('download.html', allname=Foder_Name, name=Files_Name)

    # path ='data/word_stop_flag.josn'
    # try:
    #     # 尝试打开文件并发送
    #     return send_file(path, as_attachment=True)
    # except PermissionError:
    #     return '文件权限错误，无法访问', 403
    # except Exception as e:
    #     return f'发生错误：{str(e)}', 500


@app.route('/downloads/<path:path>', methods=['GET', 'POST'])
def downloads(path):
    """ 下载 """
    """
        重写download方法，根据前端点击的文件传送过来的path，下载文件
		send_from_directory：用于下载文件
		flask.send_from_directory(所有文件的存储目录，相对于要下载的目录的文件名，as_attachment：设置为True是否要发送带有标题的文件)
    """
    if session.get("username") == 'admin':
        return send_from_directory(app.config['DATA_FOLDER'], path, as_attachment=True)
    else:
        return "非管理员！<a href='/'>首页</a>"


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    """  文件上传  """
    if request.method == 'POST':
        # input标签中的name的属性值
        f = request.files['file']
        if session.get("username") == 'admin':
            # 拼接地址，上传地址，f.filename：直接获取文件名
            f.save(os.path.join(app.config['DATA_FOLDER'], f.filename))
            # 输出上传的文件名
            print(request.files)
            return f'文件{f.filename}上传成功!'
        else:
            return "非管理员！<a href='/'>首页</a>"

    return render_template('upload.html')


@app.route('/proxy/<path:url>/<string:name>/<string:password>', methods=['GET'])
def proxy(url, name, password):
    if session.get("username") != 'admin':
        return jsonify({'error': 'The user is not admin!'}), 400

    import requests
    auth = (name, password) if name and password else None
    response = requests.get(f'https://{url}', auth=auth)

    if response.status_code == 200:
        return Response(response.content, content_type=response.headers['Content-Type'])
    else:
        return "Failed to fetch the page", response.status_code


@app.route('/register', methods=['GET', 'POST'], endpoint='register')
def register():
    error = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        validation = request.form['validation']
        if username != 'admin':
            if validation in ('ideatech.info', 'ideatech'):
                uid, inf = swg.register_user(username)
                if uid >= 0:
                    filtered_users = User.query.filter_by(
                        user_id=uid).all()  # User.query.filter(User.username == username).first()
                    if username not in (user.username for user in filtered_users):
                        new_user = User(user_id=uid, username=username, password=password,
                                        cross=inf['cross'])
                        db.session.add(new_user)
                        db.session.commit()
                        session['username'] = username
                        return render_template("index.html")
                    elif password in (user.password for user in filtered_users):
                        session['username'] = username
                        return render_template("index.html")
                    else:
                        error = '用户已注册'
                        # return redirect(url_for('login'))
                else:
                    error = '用户已满'
            else:
                error = '验证失败'
        else:
            error = '该用户禁止注册'

    return render_template('register.html', error=error)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        uid, inf = swg.get_user(username)
        flag = 0
        if uid >= 0:
            filtered_users = User.query.filter_by(user_id=uid).all()
            for user in filtered_users:
                if user.username == username:
                    flag |= 1 << 1
                    if user.password == password:
                        flag |= 1 << 2
                        break
        else:
            filtered_users = User.query.filter_by(username=username).all()
            for user in filtered_users:
                if user.password == password:
                    swg.register_user(username)
                    flag |= 1 << 2
                    break
        if flag & (1 << 2) != 0:
            session['username'] = username  # 将用户名存储在会话中
            return render_template("index.html")  # redirect(url_for(f'/user/{username}'))
        if flag & (1 << 1) != 0:
            return render_template('login.html', error='Invalid password')
        return redirect(url_for('register', error='Invalid username'))

    return render_template('login.html')


@app.route('/logout', endpoint='logout')
def logout():
    # remove the username from the session if it's there
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/home', methods=['GET'], endpoint='home')
def home():
    # 获取当前登录用户的用户名
    username = session.get("username")
    if username:
        paragraphs = [
            '1 专利',
            '2 融资',
            '3 专利与融资'
        ]
        return render_template("index.html", title='Home', data=paragraphs)
    # username = request.cookies.get('username')
    # if 'username' in session:
    #     return f'Welcome, Logged in as {session["username"]}'
    ##<a href='https://ideatech.info/'>易得融信</a>
    return redirect("/login")


@app.route('/', methods=['GET'], endpoint='index')
def index():
    return render_template('index.html', title='Web App')


if __name__ == "__main__":
    graph = None
    if DEBUG_MODE:
        # word_data = pd.read_excel('data/patent_doc_cut_word.xlsx', index_col=0, nrows=1000)
        # word_data = word_data[(word_data['w2v'] > 0)].rename(
        #     columns={'词语': 'word',
        #              '序号': 'index',
        #              'cut_max_count': 'cmc',
        #              'doc_max_count': 'dmc',
        #              '标题 (中文)': 'patent_title'}).sort_values(
        #     ['index', 'dmc', 'cmc', 'tf'], ascending=[True, False, False, False])
        #
        # swg.load(word_data)
        # swg.reset()
        swg.register_user('test')
        # from w2v_neo4j_net import *
        # from vec_neo4j_net import VRelationships
        # from matrix_neo4j_net import XYRelationships
        # wrs = WordRelationships()
        # vr = VRelationships()
        # xyr = XYRelationships()
        # wrs.load(graph=None,
        #          wo=Word2Vec.load("data/patent_w2v.model"),
        #          wo_sg=Word2Vec.load("data/patent_w2v_sg.model"))
        # xyr.load(graph=None, data=pd.read_excel('data/co_cosine_sim_3.xlsx', index_col=0))
        # vr.load(graph=None, data=pd.read_parquet('data/patent_co_vec_23.parquet'))
    else:
        try:
            graph = Graph(f"bolt://{Config.NEO4J_HOST}:7687", auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD))
        except:
            print('neo4j graph connection unavailable')

        log = logging.getLogger('werkzeug')
        log.addFilter(IgnoreHeadRequestsFilter())

        # Config.QDRANT_HOST = '47.110.156.41'

    client = QdrantClient(host=Config.QDRANT_HOST, grpc_port=6334, prefer_grpc=True)
    vdr.load(graph, client, collection_name='专利_w2v', prefix='Word',
             match_values=['医疗健康', '先进制造', '传统制造', '金融', '金融科技'])

    vdr.append(client, collection_name='专利_w2v_188_37', prefix='Word', match_values=['all'])
    vdr.append(client, collection_name='企业_w2v_lda', prefix='Co',
               match_values=['先进制造', '医疗健康', '金融科技', 'all'])

    db.init_app(app)  # Flask对象与QLAlchemy()进行绑定
    with app.app_context():  # init_db
        db.create_all()  # 根据继承的Model的类创建表，如果表名已经存在，则不创建也不更改

        table = pd.read_sql(f'select * from users', con=db.engine).set_index('id')
        table = table[['user_id', 'username', 'cross']].rename(
            columns={'user_id': 'uid', 'username': 'name'})
        table['readn'] = 0
        table['stopn'] = 0
        swg.user_data = table.to_dict(orient='records')

        table = pd.read_sql(f'select * from word_tf_df_sc', con=db.engine).set_index('word')
        swg.load(table)
        table = pd.read_sql(f'select word,read_flag,stop_flag from stop_words', con=db.engine).set_index('word')
        table.columns = ['阅读标记', '停用标记']
        swg.word_data.update(table)

        swg.reset_user_data()
        swg.get_stop_words(-1)

    app.run(debug=DEBUG_MODE, port=3300, host='0.0.0.0')

    if not DEBUG_MODE:
        swg.save_flag()

    with app.app_context():
        # 在应用上下文中执行数据库操作
        db.session.commit()
        db.session.close()
