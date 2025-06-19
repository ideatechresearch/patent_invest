import numpy as np
import pandas as pd
import os, re
import psutil
from generates import call_ollama
from utils import extract_json_from_string, fuzzy_match_template, save_dictjson, load_dictjson

os.environ["LOKY_MAX_CPU_COUNT"] = str(psutil.cpu_count(logical=False))


def clean_applynote(d1, save_path):
    topn_ratio = 0.85
    topn_min = 12
    print(d1.columns, d1.shape[0], d1.yd_applynote.str.endswith(';').sum())

    d1 = d1.dropna(subset="yd_applynote").reset_index(drop=True)

    def clean_date_text(text):
        """
        去除开头类似 '2020-01-09;10:44;' 的时间戳字段，保留后面的实际备注内容。
        """
        if not isinstance(text, str):
            return text
        # 匹配开头形如 "YYYY-MM-DD;HH:MM;" 格式，可能有多个连续时间字段
        pattern = r'^(?:\d{4}-\d{2}-\d{2};\d{2}:\d{2};)+'
        return re.sub(pattern, '', text).strip()

    d1['yd_applynote'] = d1['yd_applynote'].apply(clean_date_text)
    d1["yd_applynote"] = d1["yd_applynote"].apply(
        lambda x: x.strip() + ";" if isinstance(x, str) and not x.endswith(";") else x.strip())

    # d1["applynote"] = d1["yd_applynote"].str.rstrip(";").str.split(";")
    # d1s = d1.explode("applynote").reset_index(drop=True)
    # # d1s = d1s.to_frame()
    # d1s["applynote"] = d1s["applynote"].str.strip().str.rstrip(';,.，。！？!').replace("", np.nan)
    # d1s = d1s.dropna(subset="applynote")
    # counts = d1s["applynote"].value_counts()
    d1["applynote_list"] = d1["yd_applynote"].str.rstrip(";").str.split(";")
    flat_notes = pd.Series(
        [note.rstrip(';,.，。！？!').strip() for sublist in d1["applynote_list"] for note in sublist if note.strip() != ""])
    print(len(flat_notes))
    counts = flat_notes.value_counts()
    cumulative = np.cumsum(counts.values) / counts.sum()
    cutoff_index = np.where(cumulative <= topn_ratio)[0][-1]  # int(np.sum(cumulative < topn_ratio))

    topn = max(topn_min, counts.iloc[cutoff_index])
    top_questions = counts[counts >= topn].index.tolist()
    print(cutoff_index, topn)
    print(top_questions)
    # top_questions.drop(['资料不全', '资料不齐', '重复预约'])
    # 205 27,261 10,260 8

    mask = d1["yd_applynote"].str.contains('标准电核105001电核不通过')  # .astype(int)
    print(mask.sum(), np.sum(~mask))
    d1.loc[mask, 'class'] = "标准电核105001电核不通过"
    d1.loc[mask, 'applynote'] = d1.loc[mask, "yd_applynote"].str.extract(r"详情：(.*)")[0]

    def extract_class_note(segments, top_qs):
        segments = [seg.rstrip(';,.，。！？!').strip() for seg in segments if seg.strip()]
        if not segments:
            return {'class': np.nan, 'applynote': ''}
        for q in top_qs:  # 按 top_qs 的顺序优先
            if q in segments:  # 完全匹配 segment
                remain = [s for s in segments if s != q]
                if len(segments) > 1 and not any(s in top_qs for s in remain):
                    pass
                else:
                    remain = segments

                return {'class': q, 'applynote': ';'.join(remain).strip()}

        return {'class': np.nan, 'applynote': ';'.join(segments).strip()}

    def extract_class_note_from_text(full_text, top_qs):
        for q in top_qs:
            if q in full_text:
                segments = full_text.split(q, 1)
                remain = segments[1].split(";", 1)
                return {'class': q, 'applynote': remain[1].strip() if len(remain) > 1 else ''}

        return {'class': np.nan, 'applynote': full_text}

    # res = d1.loc[~mask, 'yd_applynote'].apply(lambda x: extract_class_note_from_text(x, top_questions))

    res = d1.loc[~mask, 'applynote_list'].apply(lambda x: extract_class_note(x, top_questions))
    d1.loc[~mask, ['class', 'applynote']] = pd.DataFrame(res.tolist(), index=res.index)

    # print(d1.loc[~mask, ['yd_applynote', 'class', 'applynote']])

    d1 = d1[~d1["applynote"].fillna("").astype(str).str.fullmatch(r"\d+")]
    print((d1['applynote'] == '').sum())
    d1 = d1.dropna(subset=['applynote']).copy().reset_index(drop=True)

    d1['details'] = d1['applynote'].str.rstrip(";").str.split(";")
    d1s = d1.explode('details')
    d1s['details'] = d1s['details'].str.strip().str.rstrip(';,.，。！？!').replace("", np.nan)
    d1s = d1s.dropna(subset=['details']).reset_index(drop=True)
    print(d1s.shape[0])
    d1s.to_excel(save_path)

    d1n = d1s.details.unique()  # 44812
    counts = d1s.details.value_counts()
    top_questions = counts[counts >= 30].index.tolist()
    print(top_questions)
    return d1s, d1n


def intra_cluster_similarity_mean(df, text_embeddings, deduplicate=True):
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_means = np.full(len(df), np.nan)

    for cluster_id, group in df.groupby('cluster'):
        if cluster_id == -1 or len(group) <= 1:
            continue

        if deduplicate:
            # 去重后的嵌入索引
            unique_indices = group['emb_index'].unique()
            if len(unique_indices) <= 1:
                continue
            embs = text_embeddings[unique_indices]
            sim_matrix = cosine_similarity(embs)
            np.fill_diagonal(sim_matrix, np.nan)
            sim_means = np.nanmean(sim_matrix, axis=1)
            # 将相似度映射回原始 group
            emb_idx_to_sim = dict(zip(unique_indices, sim_means))
            similarity_means[group.index] = group['emb_index'].map(emb_idx_to_sim)

        else:
            indices = group['emb_index'].values
            if len(indices) <= 1:
                continue
            embs = text_embeddings[indices]
            sim_matrix = cosine_similarity(embs)
            for i, idx in enumerate(group.index):
                sim_row = np.delete(sim_matrix[i], i)
                similarity_means[idx] = sim_row.mean()

    return similarity_means


def emb_clusterer(df):
    try:
        from sentence_transformers import SentenceTransformer
        import hdbscan
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
        from sklearn.metrics.pairwise import cosine_distances
        import sklearn.metrics as sm
        d1n = df.details.unique()
        details_map = {val: idx for idx, val in enumerate(d1n)}
        df['emb_index'] = df.details.map(details_map)
        counts = df.details.value_counts()
        print(len(d1n))
        model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        text_embeddings = model.encode(d1n.tolist(), normalize_embeddings=True, show_progress_bar=True)
        print(text_embeddings.shape)
        np.savez('data/embeddings_开立银行账户备注陈述详情_bge-large-zh.npz', embeddings=text_embeddings, details=d1n)

        offset = 0
        offset2 = 0
        cluster_results = []
        df_res = pd.DataFrame()
        cos_sim_th = 0.85
        drop_ratio_sim = 0.05  # 比如丢掉相似度最低的 5%
        drop_ratio_dist = 0.005  # 丢掉距离最大 0.5%
        topn_ratio = 0.85
        for t, group in df.groupby('type'):
            idx = group['emb_index'].values
            sub_embeddings = normalize(text_embeddings[idx])
            k = {'开户': 200, '变更': 250, '销户': 250}.get(t, 100)
            n = {'开户': 27, '变更': 10, '销户': 8}.get(t, 10)
            kmeans_k = KMeans(n_clusters=k, random_state=42)
            kmeans_k.fit(sub_embeddings)

            idx2 = group['emb_index'].unique()
            sub_embeddings2 = normalize(text_embeddings[idx2]).astype('float64')
            clusterer = hdbscan.HDBSCAN(
                metric='cosine',
                algorithm='generic',
                min_cluster_size=7,
                min_samples=3,
                core_dist_n_jobs=-1
            )
            sub_labels = clusterer.fit_predict(sub_embeddings2)

            d1n_cluster = group[
                ['yd_applynote', 'yd_type', 'yd_bill_type', 'class', 'details', 'emb_index']].copy().reset_index(
                drop=True)  # .to_frame()
            d1n_cluster['hdbscan_cluster'] = d1n_cluster['emb_index'].map(dict(zip(idx2, sub_labels + offset2)))
            d1n_cluster['hdbscan_probability'] = d1n_cluster['emb_index'].map(dict(zip(idx2, clusterer.probabilities_)))

            d1n_cluster['cluster'] = kmeans_k.labels_ + offset
            d1n_cluster['intra_cluster_distances'] = np.linalg.norm(
                sub_embeddings - kmeans_k.cluster_centers_[kmeans_k.labels_], axis=1)  # 每个样本到簇中心的距离
            d1n_cluster['counts'] = d1n_cluster['details'].map(d1n_cluster['details'].value_counts())
            d1n_cluster['cluster_counts'] = d1n_cluster['cluster'].map(d1n_cluster['cluster'].value_counts())
            # d1n_cluster['is_representative'] = (d1n_cluster.groupby('cluster')['intra_cluster_distances'].rank(method='first', ascending=True) <= n).astype(int)
            d1n_cluster["intra_cluster_cos_sim"] = intra_cluster_similarity_mean(d1n_cluster,
                                                                                 text_embeddings)  # 计算每个样本与同簇其他样本的余弦相似度均值
            d1n_cluster.sort_values(
                ['cluster_counts', 'counts', 'cluster', 'intra_cluster_distances', 'intra_cluster_cos_sim'],
                ascending=[False, False, True, True, False, ]).to_excel(f'data/预约开户问题聚类_{t}_km{k}.xlsx')
            offset += kmeans_k.labels_.max() + 1  # 避免簇编号重复
            offset2 += sub_labels.max() + 1

            calinski = sm.calinski_harabasz_score(sub_embeddings, kmeans_k.labels_)
            print(f"[{t}] Calinski-Harabasz 得分: {calinski:.2f}")
            print(offset, offset2)

            dk = get_topn_by_cluster(d1n_cluster, cos_sim_th, drop_ratio_sim, drop_ratio_dist, topn_ratio)
            df_res = pd.concat([df_res, dk], ignore_index=True)

        # print(sm.normalized_mutual_info_score(df_res['hdbscan_cluster'], df_res['cluster']),
        #       sm.adjusted_rand_score(df_res['hdbscan_cluster'], df_res['cluster']))
        # 基于 HDBSCAN 清洗噪声 + KMeans 聚类.聚类一致性指标, NMI（归一化互信息）ARI（调整兰德指数）
        print(sm.normalized_mutual_info_score(df_res.loc[df_res['hdbscan_cluster'] != -1, 'hdbscan_cluster'],
                                              df_res.loc[df_res['hdbscan_cluster'] != -1, 'cluster']),
              sm.adjusted_rand_score(df_res.loc[df_res['hdbscan_cluster'] != -1, 'hdbscan_cluster'],
                                     df_res.loc[df_res['hdbscan_cluster'] != -1, 'cluster']))
        return df_res
    except Exception as e:
        print("聚类失败:", str(e))
        return None


def initial_centroid(x, K=3):
    """
    手动实现的 k-means++ 初始聚类中心选择算法，提高初始质心的分散性，降低局部最优的风险，提高聚类质量与收敛速度。
    用于在进行 KMeans 聚类前 更优地选取初始质心（centroids），从而避免随机初始化带来的聚类效果不稳定问题。
    :param x:
    :param K:
    :return:
    centroid=initial_centroid(X, K=3),centroid.shape
    kmeans = KMeans(n_clusters=K, init=centroid, n_init=1) #n_init=1 是因为已经提供了初始化质心，不需要再随机初始化多次。
    """
    c1_idx = int(np.random.uniform(0, len(x)))  # Step 1: 随机选取一个样本作为第一个中心,随机选取第一个中心点，随机选择一个样本作为第一个聚类中心
    centroid = x[c1_idx].reshape(1, -1)  # choice the first center for cluster.
    k = 1
    n = x.shape[0]  # 样本总数

    while k < K:  # Step 2: 选出剩下的 K-1 个中心
        d2 = []
        for i in range(n):
            # 计算样本 x[i] 到所有已选质心的欧氏距离平方
            subs = centroid - x[i, :]  # D(x) = (x_1, y_1) - (x, y)
            dimension2 = np.power(subs, 2)  # D(x)^2
            dimension_s = np.sum(dimension2, axis=1)
            d2.append(np.min(dimension_s))  # 取该样本到“最近一个”质心的距离平方

        # Step 3: 选择距离最远的样本作为下一个质心
        new_c_idx = np.argmax(d2)
        centroid = np.vstack([centroid, x[new_c_idx]])
        k += 1

    return centroid  # 返回 K 个初始化质心


def test_emb_kmeans(text_embeddings, K: list = None):
    '''
    轮廓系数（Silhouette Score）衡量一个样本和其同簇样本的相似度是否高于与其他簇的样本的相似度。
    方差比指数（Calinski-Harabasz）度量类间距离和类内距离的比值，越大越好。
    Davies-Bouldin 指数,类间距离与类内紧密度的比值，越小越好。
    :param text_embeddings:
    :param K:
    :return:
    '''
    from sklearn.cluster import KMeans
    import sklearn.metrics as sm
    if not K:
        K = range(50, 1000, 50)
    sse = []
    calinski_scores = []
    davies_scores = []
    for k in K:
        kmeans_k = KMeans(n_clusters=k, random_state=42)
        kmeans_k.fit(text_embeddings)
        sse.append(kmeans_k.inertia_)
        calinski = sm.calinski_harabasz_score(text_embeddings, kmeans_k.labels_)
        calinski_scores.append(calinski)
        davies = sm.davies_bouldin_score(text_embeddings, kmeans_k.labels_)
        davies_scores.append(davies)
        # silhouette=sm.silhouette_score(text_embeddings, kmeans_k.labels_)#silhouette score 往往分数差异不大，难以明确选出最优簇数。原因在于高维空间距离度量的“维度灾难”，以及 embedding 本身的分布特性。
        print(f"K={k}:  Calinski-Harabasz={calinski:.3f}, Davies-Bouldin={davies:.3f}")

    return sse, calinski_scores, davies_scores


def get_topn_by_cluster(df, cos_sim_th=0.85, drop_ratio_sim=0.05, drop_ratio_dist=0.005, topn_ratio=0.85):
    sim_threshold = df['intra_cluster_cos_sim'].quantile(drop_ratio_sim)
    dist_threshold = df['intra_cluster_distances'].quantile(1 - drop_ratio_dist)
    counts = df['emb_index'].value_counts()
    cumulative = np.cumsum(counts.values) / counts.sum()
    cutoff_index = int(np.sum(cumulative < topn_ratio))
    topn = counts.iloc[cutoff_index]
    print(f"阈值：sim < {sim_threshold:.4f}, dist > {dist_threshold:.4f},{cutoff_index}, {topn}")

    mask = ((df['intra_cluster_cos_sim'].fillna(1) >= min(cos_sim_th, sim_threshold)) & (
            df['intra_cluster_distances'] <= dist_threshold))
    df['is_noise'] = (~(mask & (df['cluster'] != -1))).astype(int)
    print(f"保留占比: {mask.sum() / df.shape[0]:.4f}")
    sorted_filtered = df[df['is_noise'] == 0].copy().sort_values(
        by=['cluster', 'counts', 'intra_cluster_distances', 'intra_cluster_cos_sim'],
        ascending=[True, False, True, False]
    )
    # 每个 cluster + emb_index 保留一个代表样本
    deduped = sorted_filtered.groupby(['cluster', 'emb_index'], as_index=False).first()
    # 在每个 cluster 中选 top-N
    topn_emb_ids = deduped.groupby('cluster')['emb_index'].head(topn)

    df['is_topn'] = df['emb_index'].isin(topn_emb_ids).astype(int)
    print(df.loc[df['is_topn'] == 1, 'emb_index'].nunique() / df['emb_index'].nunique(),
          df.loc[df['is_topn'] == 0, 'emb_index'].nunique() / df['emb_index'].nunique())

    return df


def cluster_summary():
    df_res = pd.DataFrame()
    cos_sim_th = 0.85
    drop_ratio_sim = 0.05  # 比如丢掉相似度最低的 5%
    drop_ratio_dist = 0.005  # 丢掉距离最大 0.5%
    topn_ratio = 0.85

    for path in ['data/预约开户问题聚类_开户_km200.xlsx', 'data/预约开户问题聚类_变更_km250.xlsx',
                 'data/预约开户问题聚类_销户_km250.xlsx']:
        dk = get_topn_by_cluster(pd.read_excel(path, index_col=0), cos_sim_th, drop_ratio_sim, drop_ratio_dist,
                                 topn_ratio)
        df_res = pd.concat([df_res, dk], ignore_index=True)

    print(df_res.is_noise.sum() / df_res.shape[0])  # 0.0440514611609708
    df_res.to_excel('data/开立银行账户备注陈述详情_km_all.xlsx')

    noise_stats = df_res.groupby(['yd_bill_type', 'cluster'])['is_noise'].mean().reset_index(name='noise_ratio')
    print(noise_stats['noise_ratio'].mean(), (noise_stats['noise_ratio'] > 0.5).sum())
    details_calss = df_res[df_res['is_noise'] == 0].groupby(['yd_bill_type', 'cluster'])['class'].unique().reset_index()
    details_topn = df_res[df_res['is_topn'] == 1].groupby(['yd_bill_type', 'cluster'])['details'].unique().reset_index()
    details_noise = df_res[df_res['is_noise'] == 1].groupby(['yd_bill_type', 'cluster'])[
        'details'].unique().reset_index(name='noise_details')

    summary = noise_stats.merge(details_calss, on=['yd_bill_type', 'cluster'], how='left')
    summary = summary.merge(details_topn, on=['yd_bill_type', 'cluster'], how='left')
    summary = summary.merge(details_noise, on=['yd_bill_type', 'cluster'], how='left')

    cluster_noise_summary = summary.sort_values(['noise_ratio', 'cluster'], ascending=True)
    cluster_noise_summary['class'] = cluster_noise_summary['class'].dropna().apply(
        lambda x: ';'.join([str(i) for i in x if pd.notna(i)]))
    cluster_noise_summary.details = cluster_noise_summary.details.dropna().apply(lambda x: ';'.join(x))
    cluster_noise_summary.noise_details = cluster_noise_summary.noise_details.dropna().apply(lambda x: ';'.join(x))
    cluster_noise_summary.to_excel('data/cluster_summary_topn.xlsx', index=False)

    return df_res, summary


async def summary_description(cluster_noise_summary):
    res = []
    for _, row in cluster_noise_summary.iterrows():
        if pd.isna(row['details']):
            continue
        sysc = f''''
        你是一名熟悉银行对公开户流程的智能归因专家。现在有一些{row['bill_type']}失败被退回的原因描述，这些描述来源于 embedding 语义聚类后的文本集合，由不同业务人员填写，内容可能存在表述模糊、语义重复、句式差异等问题。

        请你根据以下要求对聚类结果中的文本进行标准化归纳：

        【目标】：
        - 从聚类内的多个相似表述中，提炼出一个语义清晰、准确、完整的标准表达，作为该聚类的“代表问题”
        - 去除重复、冗余或非关键信息，保留核心问题描述
        - 保证语义通顺、专业、简洁，便于归因分析和后续建模使用

        【清洗与归纳规则】：
        1. 如果原始表达过于简略，请结合银行常见业务背景进行合理补全；
        2. 如果原始表达含有重复或客套语句，请去除无效部分，仅保留问题核心；
        3. 多条表达语义类似但用词不同时，请统一同义词并归并为一句标准问题；
        4. 输出应为一句完整、准确、清晰的问题句；
        5. 严禁输出“请注意”、“谢谢”、“重新提交”等非问题本身的内容。

        【输出格式】：
        仅输出本组文本的清洗归一化结果（即该类问题的标准表达），不要附加解释说明、编号或原始文本。

        【示例参考】：
        输入：
        客户未接电话；客户电话无法接通；客户预约电话无人接听；多次拨打客户电话未接通
        输出：
        客户预约时预留的联系电话无法接通，导致无法联系客户完成开户确认流程

        现在请根据以上规则处理以下聚类文本，输出本聚类的标准归因问题表述：
        返回格式：{{"description": "标准问题句"}}
        '''
        messages = [{
            "role": "system", "content": sysc},
            {"role": "user", "content": str(row['details'])}
        ]
        try:
            result = await call_ollama('你好啊', messages=messages, model_name="qwen3:14b", host='10.168.1.10',
                                       time_out=300, stream=False)
            # print(result)
            content = result.get('message', {}).get('content', '')
            data = extract_json_from_string(content.split("</think>")[-1])
            print(_, row['cluster'], data)
            res.append(
                {'idx': _, 'cluster': row['cluster'], 'content': content, 'description': data.get('description')})
        except Exception as e:
            print(e)

    # results_df = pd.DataFrame(res).set_index('idx')
    # merged_df = cluster_noise_summary.merge(
    #     results_df[['content', 'description']],
    #     how='left', left_index=True, right_index=True,
    #     # right_on='idx'
    # )
    return res


def nested_dict_structure(df):
    from collections import defaultdict
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for _, row in df.iterrows():
        result[row['一级类']][row['二级类']][row['三级类']].append(row['template'])
    return result


def format_entry(row):
    return f"分类路径：{row['一级类']} > {row['二级类']} > {row['三级类']}\n语义模板：{row['template']}"


def restore_templates_groupby(df):
    result = []
    for level1, df_l1 in df.groupby('一级类'):
        level1_node = {'一级类': level1, 'children': []}
        for level2, df_l2 in df_l1.groupby('二级类'):
            level2_node = {'二级类': level2, 'children': []}
            for level3, df_l3 in df_l2.groupby('三级类'):
                templates = df_l3['template'].tolist()
                level3_node = {'三级类': level3, 'templates': templates}
                level2_node['children'].append(level3_node)
            level1_node['children'].append(level2_node)
        result.append(level1_node)
    return result


def make_template(data: list):
    def flatten_templates(tree, level1=None, level2=None):
        rows = []
        # 有语义模板但还没进入三级层级时，说明是二级类的模板
        if 'semantic_templates' in tree:
            if level1 and not level2:
                for template in tree['semantic_templates']:
                    rows.append({
                        '一级类': level1,
                        '二级类': tree['name'],
                        '三级类': None,
                        'template': template
                    })
            elif level1 and level2:
                for template in tree['semantic_templates']:
                    rows.append({
                        '一级类': level1,
                        '二级类': level2,
                        '三级类': tree['name'],
                        'template': template
                    })
        elif 'children' in tree:
            current_level = tree['name']
            for child in tree['children']:
                if level1 is None:
                    rows.extend(flatten_templates(child, level1=current_level))
                elif level2 is None:
                    rows.extend(flatten_templates(child, level1=level1, level2=current_level))
                else:
                    # 超过三级则不处理或忽略
                    pass

        return rows

    rows = []
    for d in data:
        rows += flatten_templates(d)

    df_template = pd.DataFrame(rows)
    df_template.to_excel('data/semantic_templates.xlsx', index=False)
    print(len(data), df_template.一级类.unique(), df_template['template'].nunique(), df_template.shape)
    json_structure = restore_templates_groupby(df_template)
    save_dictjson(json_structure, 'data/semantic_templates_structure.json')
    return df_template, json_structure


async def structure_sample_applynote(cx, df_sample, df_template, json_structure, model='qwen:qwen3-32b',
                                     save_name='data/sample_开户过程分类优化8.xlsx'):
    '''
    update 是“已有路径 + 补模板”
    insert 是“路径都没有 + 创建路径 + 补模板”
    '''
    from agents.ai_prompt import System_content
    sys_prompt1 = System_content['103'].format(templates_list=df_template['template'].to_list())
    sys_prompt3 = System_content['105'].format(structure=json_structure)
    print(len(sys_prompt1), len(sys_prompt3))

    with open("data/开户模板分类提示.txt", "w", encoding="utf-8") as file:
        file.write(sys_prompt1)
    with open("data/开户模板分类提示优化.txt", "w", encoding="utf-8") as file:
        file.write(sys_prompt3)

    def build_prompt_template(user_question, user_q) -> str:
        prompt = f"""以下是用户原始输入：{user_q}
    清洗后的标准化问题，以增强理解：{user_question}

    请你思考后，基于以上信息，以原始输入为主，并参考已有模板体系进行判断，仅返回符合 JSON 格式的最合适模板。"""
        return prompt

    def build_prompt_new(user_question, user_q) -> str:
        prompt = f"""以下是用户原始输入为：{user_q}
    用户清洗后的标准化问题，以增强理解：{user_question}

    请你思考后，基于以上信息，以原始输入为主，并参考已有模板体系，结合各种情况进行判断，仅返回符合 JSON 格式的最合适分类路径。"""
        return prompt

    usr_p = df_sample.apply(lambda row: build_prompt_template(row['清洗后句子'], row['原始句子']), axis=1).to_list()
    df_sample["model_response"] = await cx.chat_run(rows=usr_p, system=sys_prompt1, model=model)
    error_mask = df_sample["model_response"].str.contains('error')
    print(error_mask.sum())

    match_df = pd.json_normalize(df_sample["model_response"].map(extract_json_from_string))
    df_sample['match'] = match_df['match']

    mask = df_sample['match'] != 'null'
    df_sample.loc[mask, 'matched_template'] = df_sample.loc[mask, 'match'].apply(
        lambda x: fuzzy_match_template(x, df_template['template'].tolist()))

    error_mask = ((df_sample['match'] != 'null') & df_sample['matched_template'].isna())
    if error_mask.sum() > 0:
        usr_p = df_sample[error_mask].apply(lambda row: build_prompt_template(row['清洗后句子'], row['原始句子']),
                                            axis=1).to_list()
        df_sample.loc[error_mask, "model_response"] = await cx.chat_run(rows=usr_p, system=sys_prompt1, model=model)
        print(df_sample["model_response"].str.contains('error').sum())
        parsed_series = pd.json_normalize(
            df_sample.loc[error_mask, "model_response"].map(extract_json_from_string)).iloc[:, 0]
        df_sample.loc[error_mask, 'match'] = parsed_series.values

    df_sample.to_excel('data/sample_开户过程分类优化8_tmp.xlsx')
    merged_df = df_sample.merge(df_template, left_on='matched_template', right_on='template', how='left')
    merged_df.loc[merged_df['match'] == 'null', 'match'] = None
    print(merged_df.shape)
    try:
        mask = merged_df['一级类'].isna()
        usr_p = merged_df[mask].apply(lambda row: build_prompt_new(row['清洗后句子'], row['原始句子']),
                                      axis=1).to_list()
        print(len(usr_p))
        merged_df.loc[mask, "model_response3"] = await cx.chat_run(rows=usr_p, system=sys_prompt3, model=model)
        print(merged_df.loc[mask, "model_response3"].str.contains('error').sum())

        parsed_df = pd.json_normalize(merged_df["model_response3"].map(extract_json_from_string))
        target_indices = parsed_df.index
        for col in parsed_df.columns:
            merged_df.loc[target_indices, col] = parsed_df[col].values

    except Exception as e:
        print(e)

    finally:
        merged_df.to_excel(save_name)
        return merged_df


async def structure_sample_applynote_api(df_sample, df_template, host='127.0.0.1', model='qwen:qwen3-32b'):
    '''
    update 是“已有路径 + 补模板”
    insert 是“路径都没有 + 创建路径 + 补模板”
    '''
    import aigc
    aigc.AIGC_HOST = host
    cx = aigc.Local_Aigc(host=aigc.AIGC_HOST, use_sync=False, time_out=300)

    from agents.ai_prompt import System_content
    json_structure = restore_templates_groupby(df_template)
    sys_prompt1 = System_content['103'].format(templates_list=df_template['template'].to_list())
    sys_prompt3 = System_content['105'].format(structure=json_structure)
    print(len(sys_prompt1), len(sys_prompt3))

    def build_prompt_template(user_question, user_q) -> str:
        prompt = f"""以下是用户原始输入：{user_q}
    清洗后的标准化问题，以增强理解：{user_question}

    请你思考后，基于以上信息，以原始输入为主，并参考已有模板体系进行判断，仅返回符合 JSON 格式的最合适模板。"""
        return prompt

    def build_prompt_new(user_question, user_q) -> str:
        prompt = f"""以下是用户原始输入为：{user_q}
    用户清洗后的标准化问题，以增强理解：{user_question}

    请你思考后，基于以上信息，以原始输入为主，并参考已有模板体系，结合各种情况进行判断，仅返回符合 JSON 格式的最合适分类路径。"""
        return prompt

    usr_p = df_sample.apply(lambda row: build_prompt_template(row['清洗后句子'], row['原始句子']), axis=1).to_list()
    df_sample["model_response"] = await cx.chat_run(rows=usr_p, system=sys_prompt1, model=model)
    error_mask = df_sample["model_response"].str.contains('error')
    print(error_mask.sum())

    match_df = pd.json_normalize(df_sample["model_response"].map(extract_json_from_string))
    df_sample['match'] = match_df['match']

    mask = df_sample['match'] != 'null'
    df_sample.loc[mask, 'matched_template'] = df_sample.loc[mask, 'match'].apply(
        lambda x: fuzzy_match_template(x, df_template['template'].tolist()))

    error_mask = ((df_sample['match'] != 'null') & df_sample['matched_template'].isna())
    if error_mask.sum() > 0:
        usr_p = df_sample[error_mask].apply(lambda row: build_prompt_template(row['清洗后句子'], row['原始句子']),
                                            axis=1).to_list()
        df_sample.loc[error_mask, "model_response"] = await cx.chat_run(rows=usr_p, system=sys_prompt1, model=model)
        print(df_sample["model_response"].str.contains('error').sum())
        parsed_series = pd.json_normalize(
            df_sample.loc[error_mask, "model_response"].map(extract_json_from_string)).iloc[:, 0]
        df_sample.loc[error_mask, 'match'] = parsed_series.values

    merged_df = df_sample.merge(df_template, left_on='matched_template', right_on='template', how='left')
    merged_df.loc[merged_df['match'] == 'null', 'match'] = None
    mask = merged_df['一级类'].isna()
    print(merged_df.shape)
    try:
        usr_p = merged_df[mask].apply(lambda row: build_prompt_new(row['清洗后句子'], row['原始句子']),
                                      axis=1).to_list()
        print(len(usr_p))
        merged_df.loc[mask, "model_response3"] = await cx.chat_run(rows=usr_p, system=sys_prompt3, model=model)
        print(merged_df.loc[mask, "model_response3"].str.contains('error').sum())

        parsed_df = pd.json_normalize(merged_df["model_response3"].map(extract_json_from_string))
        target_indices = parsed_df.index
        for col in parsed_df.columns:
            merged_df.loc[target_indices, col] = parsed_df[col].values

    except Exception as e:
        print(e)

    return merged_df,merged_df[mask]


def load_applynote_script():
    df_all = pd.DataFrame()
    df = pd.read_csv('data/yd_companypreopenaccountent_202505161526 开户.csv')
    df, _ = clean_applynote(df, 'data/开立银行账户备注陈述详情 开户.xlsx')
    print(len(df), len(_))
    df['type'] = '开户'
    df_all = pd.concat([df_all, df], ignore_index=True)

    df = pd.read_csv('data/yd_companypreopenaccountent_202505161531 变更.csv')
    df, _ = clean_applynote(df, 'data/开立银行账户备注陈述详情 变更.xlsx')
    print(len(df), len(_))
    df['type'] = '变更'
    df_all = pd.concat([df_all, df], ignore_index=True)

    df = pd.read_csv('data/yd_companypreopenaccountent_202505161535 销户.csv')
    df, _ = clean_applynote(df, 'data/开立银行账户备注陈述详情 销户.xlsx')
    print(len(df), len(_))
    df['type'] = '销户'
    df_all = pd.concat([df_all, df], ignore_index=True)
    df_all.to_excel('data/开立银行账户备注陈述详情_all.xlsx')
    print(len(df_all))


if __name__ == "__main__":
    # load_applynote_script()
    '''
    示例结果
    Index(['yd_applynote', 'yd_type', 'yd_bill_type'], dtype='object') 261652 150918
    205 27
    2928
    ['个人撤销', '预约信息有误', '客户取消开户', '重复提交', '营业执照影像未上传', '法人证件影像未上传', '公司其他原因', '营业执照影像文件未上传', '其他影像未上传', '已开立账户', '客户尽调未通过', '上传影像不清晰', '超期未完成法人双录', '企业不符合账户设立标准', '不符合账户设立标准', '重复预约', '其他影像资料未上传', '资料不全', '经营异常', '法人证件影像文件未上传', '其他原因', '资料不齐', '资料不齐全', '请上传原件', '客户资料不全', '预约错误', '账户性质有误', '客户未来办理', '工商与基本户信息不一致', '其他', '未开户', '账户性质错误', '重复', '法人身份证需提供原件', '无需预约', '客户重复预约', '预约重复', '客户预约错误', '账户性质选择错误', '基本户未变更', '账户性质错', '失信公告', '超时', '存在久悬账户', '另约时间', '客户重复发起', '户名有误', '完成', '有久悬账户', '信息有误', '客户预约账户类型有误,流程结束', '已开户', '预约有误', '账户类型有误,流程结束', ',流程结束', '暂不开户', '营业执照需提供原件', '客户资料不齐全', '客户未办理', '基本户开户许可证编号有误', '客户资料不齐', '退回', '未上门核实', '账户性质预约错误', '客户未到网点', '工商与人行信息不一致', '客户未到', '账户性质选错', '名称有误', '客户未到网点办理', '开户资料不全', '资料有误', '资料不完整', '重复申请', '账户类型选错,流程结束', '另约网点', '未上传影像', '客户重复发起，已完成双录', '无工商信息', '客户重复预约,流程结束', '营业执照未上传原件', '账户类型选择错误,流程结束', '基本户信息有误', '已存在基本存款账户', '营业执照非原件', '客户未来开户', '户名录入有误', '已开立过基本户', '单位名称有误', '预约账户类型错误,流程结束', '无影像资料', '双录失败', '工商经营范围变更，需基本户变更一致后提交', '未至办理', '请上传营业执照原件', '基本户信息未变更', '未上传影像资料', '客户要求退回', '账户类型错误', '企业名称有误', '企业名称请与营业执照核对一致！请核实后，重新预约，谢谢', '客户未到场', '无', '无法联系客户', '客户取消', '账户性质选择有误', '重新预约', '客户未来', '客户预约账户类型错误，重新预约,流程结束', '请关注我行微生活公众号-单位账户服务-对公预约开户完善开户信息，感谢您的配合', '有未注销的久悬账户', '预约网点有误', '请上传证件原件', '已开户成功', '客户重复提交', '预约时间有误', '名称错误', '已联系客户，客户称暂不开户，后续需要开户会再联系', '核准类账户无需预约', '公司名称有误', '预约类型错误', '营业执照和法人身份证需提供原件', '开户资料不齐全', '账户类型有误', '信息不全', '户名不符', '账户类型选择错误', '我行无现金存取业务，社保和缴税均无法办理', '未上传法人身份证件', '未提交资料', '系统故障', '未办理', '未来办理', '工商经营范围与人行登记信息不符，请联系基本户开户行变更后再提交', '存在久悬户', '预约网点错误', '在他行有久悬户未处理', '测试', '工商信息异常', '该企业存在工商经营异常', '未上传营业执照原件与法人身份证原件、请找网络好的地方上传', '工商异常', '户名错', '预约信息错误', '客户未至网点办理', '自然人校验不通过', '户名错误', '已受理预约开户并发送短信，此笔重复提交且未上传影像固退回', '单位名称与上传证件不符', '已完成开户', '有久悬户', '已开立基本户', '重复预约,流程结束', '请通过广西北部湾银行微生活办理预约，感谢您的配合', '账户类型错', '已开立基本存款账户，请核实后，重新预约，谢谢', '需上门核实', '核准类账户', '工商经营异常', '客户未到网点开户', '黑名单企业', '没有客户预约', '当日未办理', '与客户另约时间', '客户未至网点', '客户没来', '已预约', '重新预约时间', '未上传原件', '工商信息未更新', '待上门核实后重新预约', '重复录入', '开户许可证号码录入有误', '基本户经营范围未变更', '存款人有其他久悬银行结算账户,流程结束', '客户未开户', '客户另约时间', '客户提供资料不全', '工商存在经营异常记录,流程结束', '客户暂不开户', '身份证需提供原件', '客户经理未上门核实', '他行有久悬账户', '营业执照未年检', '名称不符', '基本户信息未更新', '预约类型有误', '请重新预约', '预约超时', '严重违法', '取消预约']
    274145 28230
    Index(['yd_applynote', 'yd_type', 'yd_bill_type'], dtype='object') 93226 60220
    261 10
    2586
    ['个人撤销', '预约信息有误', '重复提交', '营业执照影像未上传', '法人证件影像未上传', '其他影像未上传', '客户取消开户', '营业执照影像文件未上传', '公司其他原因', '上传影像不清晰', '客户尽调未通过', '资料不全', '重复预约', '无需预约', '其他影像资料未上传', '资料不齐', '资料不齐全', '不符合账户设立标准', '已开立账户', '客户取消变更', '客户资料不全', '账户性质错误', '法人证件影像文件未上传', '账户性质有误', '预约错误', '基本户未变更', '客户取消', '客户未来办理', '存在久悬账户', '有久悬账户', '参数为空,流程结束', '已变更', '客户取消办理', '无需变更', '账户性质错', '账户性质选择错误', '参数为空，客户重新预约,流程结束', '核准类账户无需预约', '他行有久悬账户', '客户资料不齐全', '预约重复', '账户类型有误,流程结束', '资料有误', '客户未办理', '客户预约基本户开户许可核准号有误,流程结束', '重复', '客户资料不齐', '工商信息未更新', '客户预约错误', '企业不符合账户设立标准', '未办理', '不需要预约', '开户许可证号未录入,流程结束', '有久悬', '另约时间', '客户预约信息录入不完整，重新预约,流程结束', '资料不完整', '户名有误', '客户预约账户类型有误,流程结束', '客户重复预约', '其他', '一般户', '未录入基本户开户许可证号,流程结束', '客户未到网点办理', '预约有误', '非我行账户', '印章不符', '客户改天来办理', '他行有久悬', '工商异常', '校验状态不通过，参数为空,流程结束', '一般户变更无需预约', ',流程结束', '账户性质选错', '一般户无需预约', '退回', '开户许可证号错,流程结束', '有久悬户', '开户许可证影像未上传', '基本户核准号未录入,流程结束', '存在久悬户', '核准类', '未录入开户许可证号,流程结束', '业务未办理', '变更资料不全', '取消办理', '该企业存在工商经营异常', '开户许可证不正确,流程结束', '核准类账户', '客户未到', '信息有误', '账户类型选择错误,流程结束', '账户性质不符', '公章不符', '预约账户类型错误,流程结束', '客户预约账户类型错误，重新预约,流程结束', '账户类型选错,流程结束', '在他行有久悬账户', '取消变更', '经营异常']
    101088 11328
    Index(['yd_applynote', 'yd_type', 'yd_bill_type'], dtype='object') 72996 46337
    260 8
    1118
    ['个人撤销', '预约信息有误', '重复提交', '营业执照影像未上传', '法人证件影像未上传', '营业执照影像文件未上传', '其他影像未上传', '客户取消开户', '公司其他原因', '上传影像不清晰', '重复预约', '资料不全', '客户取消销户', '客户尽调未通过', '其他影像资料未上传', '资料不齐', '法人证件影像文件未上传', '不符合账户设立标准', '客户取消', '账户性质错误', '资料不齐全', '客户资料不全', '账户性质有误', '开户许可证影像未上传', '无需预约', '账户性质错', '客户未来办理', '预约错误', '客户重复预约', '印章不符', '已开立账户', '客户取消办理', '非我行账户', '已销户', '核准类账户无需预约', '重复', '销户资料不全', '预约重复', '账户性质选择错误', '客户未到', '印鉴不符', '客户资料不齐全', '客户未办理', '取消销户', '其他', '客户资料不齐', '客户预约账户类型有误,流程结束', '预约账户类型错误,流程结束', '账户类型有误,流程结束', '未办理', '公章不符', '差开户许可证影像', '资料有误', '户名有误', '客户未到网点办理', '账户性质选错', '客户预约错误', '账户类型选择错误,流程结束', '账户类型错误', '核准类账户', '客户重复提交', '印章有误', '客户未到网点', '预约有误', '其他原因', '核准类', '另约时间', '账户类型选错,流程结束', '资料不完整', '预约网点有误', '开户许可证未上传', '企业不符合账户设立标准']
    79472 9480
    454705
    
     Index(['yd_applynote', 'yd_type', 'yd_bill_type'], dtype='object') 261652 150918
    274362
    205 27
    ['个人撤销', '预约信息有误', '客户取消开户', '重复提交', '营业执照影像未上传', '法人证件影像未上传', '公司其他原因', '营业执照影像文件未上传', '其他影像未上传', '已开立账户', '客户尽调未通过', '上传影像不清晰', '超期未完成法人双录', '企业不符合账户设立标准', '不符合账户设立标准', '重复预约', '其他影像资料未上传', '资料不全', '经营异常', '法人证件影像文件未上传', '其他原因', '资料不齐', '资料不齐全', '请上传原件', '客户资料不全', '预约错误', '账户性质有误', '客户未来办理', '工商与基本户信息不一致', '其他', '未开户', '账户性质错误', '重复', '法人身份证需提供原件', '无需预约', '客户重复预约', '预约重复', '客户预约错误', '账户性质选择错误', '基本户未变更', '账户性质错', '失信公告', '超时', '存在久悬账户', '另约时间', '客户重复发起', '户名有误', '有久悬账户', '完成', '信息有误', '已开户', '预约有误', '暂不开户', '标准电核105001电核不通过，详情：,流程结束', '营业执照需提供原件', '1', '客户资料不齐全', '客户未办理', '基本户开户许可证编号有误', '客户资料不齐', '退回', '未上门核实', '账户性质预约错误', '客户未到网点', '工商与人行信息不一致', '客户未到', '名称有误', '账户性质选错', '客户未到网点办理', '开户资料不全', '默认退票原因:标准电核105001电核不通过，详情：账户类型有误,流程结束', '资料不完整', '资料有误', '重复申请', '另约网点', '未上传影像', '客户重复发起，已完成双录', '无工商信息', '营业执照未上传原件', '已存在基本存款账户', '基本户信息有误', '户名录入有误', '客户未来开户', '营业执照非原件', '单位名称有误', '已开立过基本户', '', '无影像资料', '双录失败', '工商经营范围变更，需基本户变更一致后提交', '未至办理', '请上传营业执照原件', '基本户信息未变更', '客户要求退回', '账户类型错误', '未上传影像资料', '企业名称请与营业执照核对一致！请核实后，重新预约，谢谢', '企业名称有误', '无', '客户未到场', '无法联系客户', '账户性质选择有误', '客户取消', '客户未来', '重新预约', '有未注销的久悬账户', '预约网点有误', '请上传证件原件', '请关注我行微生活公众号-单位账户服务-对公预约开户完善开户信息，感谢您的配合', '已开户成功', '名称错误', '客户重复提交', '预约时间有误', '其他退票原因:标准电核105001电核不通过，详情：客户预约账户类型有误,流程结束', '已联系客户，客户称暂不开户，后续需要开户会再联系', '预约类型错误', '公司名称有误', '核准类账户无需预约', '营业执照和法人身份证需提供原件', '开户资料不齐全', '账户类型选择错误', '账户类型有误', '信息不全', '户名不符', '未上传法人身份证件', '未提交资料', '我行无现金存取业务，社保和缴税均无法办理', '系统故障', '未办理', '工商经营范围与人行登记信息不符，请联系基本户开户行变更后再提交', '存在久悬户', '预约网点错误', '未来办理', '在他行有久悬户未处理', '测试', '未上传营业执照原件与法人身份证原件、请找网络好的地方上传', '该企业存在工商经营异常', '工商信息异常', '客户未至网点办理', '预约信息错误', '自然人校验不通过', '工商异常', '户名错', '户名错误', '已完成开户', '已受理预约开户并发送短信，此笔重复提交且未上传影像固退回', '单位名称与上传证件不符', '请通过广西北部湾银行微生活办理预约，感谢您的配合', '已开立基本户', '默认退票原因:标准电核105001电核不通过，详情：账户类型选择错误,流程结束', '有久悬户', '账户类型错', '已开立基本存款账户，请核实后，重新预约，谢谢', '核准类账户', '标准电核105001电核不通过，详情：客户预约账户类型有误,流程结束', '需上门核实', '当日未办理', '客户未到网点开户', '黑名单企业', '工商经营异常', '与客户另约时间', '没有客户预约', '未上传原件', '客户未至网点', '客户没来', '已预约', '重新预约时间', '工商信息未更新', '基本户经营范围未变更', '重复录入', '待上门核实后重新预约', '开户许可证号码录入有误', '客户未开户', '客户提供资料不全', '默认退票原因:标准电核105001电核不通过，详情：账户类型选错,流程结束', '客户另约时间', '客户暂不开户', '身份证需提供原件', '客户经理未上门核实', '预约类型有误', '取消预约', '他行有久悬账户', '标准电核105001电核不通过，详情：客户重复预约,流程结束', '名称不符', '严重违法', '营业执照未年检', '基本户信息未更新', '请重新预约', '预约超时', '0', '客户经理上门尽职调查时预约', '默认退票原因:标准电核105001电核不通过，详情：预约账户类型错误,流程结束', '默认退票原因:标准电核105001电核不通过，详情：客户预约账户类型错误，重新预约,流程结束', '开户资料不齐', '标准电核105001电核不通过，详情：账户类型选错,流程结束', '基本户信息与工商信息不一致', '账户类型选错', '开户许可证号码有误', '基本户不唯一', '客户信息有误', '未查询到工商信息', '标准电核105001电核不通过，详情：预约账户类型错误,流程结束', '基本户核准号有误', '已开立基本账户', '账户性质录入错误', '取消预约，另约时间，16181', '账户名称有误', '人行信息有误', '重复发起', '存在久悬', '预约账户类型有误']
    2928 250255
    119
    272594
    ['个人撤销', '预约信息有误', '客户取消开户', '重复提交', '营业执照影像未上传', '法人证件影像未上传', '公司其他原因', '营业执照影像文件未上传', '其他影像未上传', '已开立账户', '客户尽调未通过', '上传影像不清晰', '超期未完成法人双录', '企业不符合账户设立标准', '不符合账户设立标准', '重复预约', '其他影像资料未上传', '资料不全', '经营异常', '法人证件影像文件未上传', '其他原因', '资料不齐', '资料不齐全', '请上传原件', '客户资料不全', '预约错误', '账户性质有误', '客户未来办理', '工商与基本户信息不一致', '其他', '未开户', '账户性质错误', '重复', '法人身份证需提供原件', '无需预约', '客户重复预约', '预约重复', '客户预约错误', '账户性质选择错误', '基本户未变更', '账户性质错', '超时', '存在久悬账户', '另约时间', '客户重复发起', '户名有误', '失信公告', '完成', '有久悬账户', '信息有误', '客户预约账户类型有误,流程结束', '预约有误', '已开户', '账户类型有误,流程结束', ',流程结束', '暂不开户', '营业执照需提供原件', '客户资料不齐全', '基本户开户许可证编号有误', '客户未办理', '客户资料不齐', '退回', '未上门核实', '账户性质预约错误', '客户未到网点', '工商与人行信息不一致', '名称有误', '账户性质选错', '客户未到', '客户未到网点办理', '开户资料不全', '资料有误', '资料不完整', '重复申请', '账户类型选错,流程结束', '另约网点', '未上传影像', '客户重复发起，已完成双录', '无工商信息', '营业执照未上传原件', '客户重复预约,流程结束', '账户类型选择错误,流程结束', '基本户信息有误', '已存在基本存款账户', '户名录入有误', '营业执照非原件', '客户未来开户', '已开立过基本户', '单位名称有误', '预约账户类型错误,流程结束', '双录失败', '无影像资料', '未至办理', '工商经营范围变更，需基本户变更一致后提交', '基本户信息未变更', '请上传营业执照原件', '企业名称有误', '企业名称请与营业执照核对一致！请核实后，重新预约，谢谢', '客户要求退回', '账户类型错误', '未上传影像资料', '无', '客户未到场', '无法联系客户', '账户性质选择有误', '客户取消', '客户未来', '客户预约账户类型错误，重新预约,流程结束', '重新预约', '有未注销的久悬账户', '请上传证件原件', '预约网点有误', '已开户成功', '请关注我行微生活公众号-单位账户服务-对公预约开户完善开户信息，感谢您的配合', '名称错误', '客户重复提交', '预约时间有误', '已联系客户，客户称暂不开户，后续需要开户会再联系', '公司名称有误', '预约类型错误', '核准类账户无需预约', '开户资料不齐全', '营业执照和法人身份证需提供原件', '账户类型有误', '户名不符', '信息不全', '账户类型选择错误', '未提交资料', '我行无现金存取业务，社保和缴税均无法办理', '未上传法人身份证件', '未办理', '系统故障', '工商经营范围与人行登记信息不符，请联系基本户开户行变更后再提交', '未来办理', '在他行有久悬户未处理', '存在久悬户', '预约网点错误', '测试', '工商信息异常', '该企业存在工商经营异常', '未上传营业执照原件与法人身份证原件、请找网络好的地方上传', '自然人校验不通过', '客户未至网点办理', '工商异常', '户名错', '预约信息错误', '户名错误', '单位名称与上传证件不符', '已完成开户', '已受理预约开户并发送短信，此笔重复提交且未上传影像固退回', '已开立基本户', '重复预约,流程结束', '有久悬户', '请通过广西北部湾银行微生活办理预约，感谢您的配合', '核准类账户', '已开立基本存款账户，请核实后，重新预约，谢谢', '账户类型错', '需上门核实', '当日未办理', '客户未到网点开户', '与客户另约时间', '没有客户预约', '工商经营异常', '客户未至网点', '重新预约时间', '工商信息未更新', '已预约', '未上传原件', '客户没来', '基本户经营范围未变更', '开户许可证号码录入有误', '重复录入', '存款人有其他久悬银行结算账户,流程结束', '待上门核实后重新预约', '客户提供资料不全', '客户另约时间', '客户未开户', '工商存在经营异常记录,流程结束', '客户暂不开户', '身份证需提供原件', '请重新预约', '名称不符', '预约类型有误', '预约超时', '基本户信息未更新', '他行有久悬账户', '取消预约', '客户经理未上门核实', '营业执照未年检']
    272594 28226
    Index(['yd_applynote', 'yd_type', 'yd_bill_type'], dtype='object') 93226 60220
    101156
    261 12
    ['个人撤销', '预约信息有误', '重复提交', '营业执照影像未上传', '法人证件影像未上传', '其他影像未上传', '营业执照影像文件未上传', '客户取消开户', '公司其他原因', '上传影像不清晰', '客户尽调未通过', '资料不全', '重复预约', '无需预约', '其他影像资料未上传', '资料不齐', '资料不齐全', '不符合账户设立标准', '已开立账户', '客户取消变更', '客户资料不全', '账户性质错误', '账户性质有误', '法人证件影像文件未上传', '预约错误', '基本户未变更', '客户取消', '客户未来办理', '存在久悬账户', '有久悬账户', '已变更', '客户取消办理', '无需变更', '账户性质错', '账户性质选择错误', '核准类账户无需预约', '他行有久悬账户', '客户资料不齐全', '预约重复', '默认退票原因:标准电核105001电核不通过，详情：参数为空,流程结束', '资料有误', '客户未办理', '重复', '默认退票原因:标准电核105001电核不通过，详情：账户类型有误,流程结束', '客户资料不齐', '工商信息未更新', '客户预约错误', '企业不符合账户设立标准', '未办理', '不需要预约', '另约时间', '资料不完整', '有久悬', '户名有误', '客户重复预约', '其他', '非我行账户', '一般户', '客户未到网点办理', '预约有误', '印章不符', '客户改天来办理', '工商异常', '他行有久悬', '一般户变更无需预约', '标准电核105001电核不通过，详情：,流程结束', '一般户无需预约', '账户性质选错', '开户许可证影像未上传', '退回', '有久悬户', '其他退票原因:标准电核105001电核不通过，详情：客户预约基本户开户许可核准号有误,流程结束', '存在久悬户', '核准类', '取消办理', '变更资料不全', '业务未办理', '客户未到', '该企业存在工商经营异常', '核准类账户', '默认退票原因:标准电核105001电核不通过，详情：开户许可证号未录入,流程结束', '信息有误', '1', '默认退票原因:标准电核105001电核不通过，详情：客户预约信息录入不完整，重新预约,流程结束', '默认退票原因:标准电核105001电核不通过，详情：未录入开户许可证号,流程结束', '其他退票原因:标准电核105001电核不通过，详情：参数为空，客户重新预约,流程结束', '1标准电核105001电核不通过，详情：参数为空,流程结束', '账户性质不符', '公章不符', '在他行有久悬账户', '标准电核105001电核不通过，详情：校验状态不通过，参数为空,流程结束', '经营异常', '默认退票原因:标准电核105001电核不通过，详情：开户许可证号错,流程结束', '取消变更', '默认退票原因:标准电核105001电核不通过，详情：未录入基本户开户许可证号,流程结束', '账户类型错误', '客户提供资料不全', '名称有误', '客户资料未带齐', '预约信息错误', '存在久悬', '其他原因', '客户未至网点办理', '客户取消预约', '账户性质预约错误', '业务取消', '重新预约', '取消', '超时', '预约类型错误', '账户性质录入错误', '资料未带齐', '公章有误', '信息未更新', '预约网点有误', '公司名称有误', '有误', '变更印鉴无需预约', '非我行客户', '账户性质选择有误', '默认退票原因:标准电核105001电核不通过，详情：开户许可证号录入有误,流程结束', '标准电核105001电核不通过，详情：参数为空，客户重新预约,流程结束', '工商经营异常', '默认退票原因:标准电核105001电核不通过，详情：账户类型选择错误,流程结束', '营业执照过期', '重复申请', '已处理', '工商信息异常', '一般账户变更无需预约', '客户有久悬账户', '不用预约', '预约类型有误', '客户存在久悬账户', '默认退票原因:标准电核105001电核不通过，详情：开户许可证不正确,流程结束', '其他退票原因:标准电核105001电核不通过，详情：客户预约账户类型有误,流程结束', '账户类型选择错误', '有久悬账户未处理', '默认退票原因:标准电核105001电核不通过，详情：客户预约账户类型错误，重新预约,流程结束', '资料缺失', '默认退票原因:标准电核105001电核不通过，详情：账户类型选错,流程结束', '未提交资料', '客户未到场', '未来办理', '', '未变更', '客户改日办理', '已电话联系', '印章有误', '客户预约有误', '客户重复提交', '未上传影像', '信息错误', '无需预约，请临柜办理', '他行存在久悬账户', '不需预约', '客户未到网点', '取消预约，影像未上传，16181', '账户类型有误', '印鉴不符', '默认退票原因:标准电核105001电核不通过，详情：预约账户类型错误,流程结束', '1标准电核105001电核不通过，详情：未录入基本开户许可证号,流程结束', '变更资料不齐全', '客户要求退回', '该业务无需预约', '默认退票原因:标准电核105001电核不通过，详情：参数为空，客户重新预约,流程结束', '预约网点错误', '非本行账户', '基本户信息未变更', '一般户久悬', '一般户变更不需要预约', '已办理', '客户未来', '提供资料不全', '默认退票原因:标准电核105001电核不通过，详情：基本户核准号未录入,流程结束', '已完成变更', '非我网点账户', '取消预约', '企业名称有误', '未至办理', '无', '其他银行有久悬账户', '标准电核105001电核不通过，详情：基本户核准号未录入,流程结束', '他行有久悬户', '取消业务', '有久悬账户，无法变更', '客户资料不完整', '名称错误', '基本账户未变更', '标准电核105001电核不通过，详情：客户预约账户类型有误,流程结束', '客户没来', '一般账户无需预约', '核准类账户不用预约', '已预约', '客户未带公章', '资料未带全', '预约时间有误', '非本网点账户', '客户未前来办理', '标准电核105001电核不通过，详情：客户预约信息录入不完整，重新预约,流程结束', '客户资料有误', '已重新预约时间', '默认退票原因:标准电核105001电核不通过，详情：核准号有误,流程结束', '标准电核105001电核不通过，详情：开户许可证不正确,流程结束', '户名不符', '客户未至网点', '账户久悬', '已受理', '未上门核实', '一般户变更无需预约，直接临柜办理', '默认退票原因:标准电核105001电核不通过，详情：基本户核准号录成专户核准号,流程结束', '久悬账户', '核准类账户不需要预约', '单位名称有误', '账户性质不对', '默认退票原因:标准电核105001电核不通过，详情：客户预约信息录入错误，重新预约,流程结束', '资料不符', '客户未来办理业务', '超期未完成法人双录', '默认退票原因:标准电核105001电核不通过，详情：名称有误,流程结束', '久悬', '影像未上传', '未在我行开户', '默认退票原因:标准电核105001电核不通过，详情：账户类型预约错误,流程结束', '核准类无需预约', '标准电核105001电核不通过，详情：未录入基本户开户许可证号,流程结束', '户名错', '重新预约时间', '法人身份证过期', '默认退票原因:标准电核105001电核不通过，详情：存款人有其他久悬银行结算账户,流程结束', '默认退票原因:标准电核105001电核不通过，详情：基本户核准号录入错误,流程结束', '标准电核105001电核不通过，详情：预约账户类型错误,流程结束', '标准电核105001电核不通过，详情：客户预约账户类型错误，重新预约,流程结束', '印鉴变更无需预约', '客户基本户未变更', '资料提供不全', '客户取消业务', '标准电核105001电核不通过，详情：客户重复预约,流程结束', '标准电核105001电核不通过，详情：客户预约信息录入错误，重新预约,流程结束', '标准电核105001电核不通过，详情：开户许可证号未录入,流程结束', '默认退票原因:标准电核105001电核不通过，详情：基本户开户许可证号录入错误,流程结束', '客户印章不符', '未到网点办理', '取消预约，另约时间，16181', '未上传影像资料', '账户名称有误']
    2586 90587
    29
    100650
    ['个人撤销', '预约信息有误', '重复提交', '营业执照影像未上传', '法人证件影像未上传', '其他影像未上传', '营业执照影像文件未上传', '客户取消开户', '公司其他原因', '上传影像不清晰', '客户尽调未通过', '资料不全', '重复预约', '无需预约', '其他影像资料未上传', '资料不齐', '资料不齐全', '不符合账户设立标准', '已开立账户', '客户取消变更', '客户资料不全', '账户性质错误', '法人证件影像文件未上传', '账户性质有误', '预约错误', '基本户未变更', '客户取消', '客户未来办理', '存在久悬账户', '有久悬账户', '参数为空,流程结束', '已变更', '客户取消办理', '无需变更', '账户性质错', '账户性质选择错误', '核准类账户无需预约', '参数为空，客户重新预约,流程结束', '他行有久悬账户', '预约重复', '客户资料不齐全', '账户类型有误,流程结束', '资料有误', '客户未办理', '客户预约基本户开户许可核准号有误,流程结束', '重复', '客户资料不齐', '工商信息未更新', '企业不符合账户设立标准', '客户预约错误', '未办理', '不需要预约', '开户许可证号未录入,流程结束', '客户预约信息录入不完整，重新预约,流程结束', '另约时间', '有久悬', '资料不完整', '客户预约账户类型有误,流程结束', '户名有误', '客户重复预约', '其他', '预约有误', '非我行账户', '未录入基本户开户许可证号,流程结束', '客户未到网点办理', '一般户', '印章不符', '客户改天来办理', '他行有久悬', '工商异常', '一般户变更无需预约', '校验状态不通过，参数为空,流程结束', '账户性质选错', '一般户无需预约', ',流程结束', '退回', '有久悬户', '开户许可证影像未上传', '开户许可证号错,流程结束', '存在久悬户', '基本户核准号未录入,流程结束', '变更资料不全', '核准类', '业务未办理', '取消办理', '未录入开户许可证号,流程结束', '客户未到', '该企业存在工商经营异常', '核准类账户', '开户许可证不正确,流程结束', '信息有误', '账户类型选择错误,流程结束', '账户性质不符', '预约账户类型错误,流程结束', '公章不符', '客户预约账户类型错误，重新预约,流程结束', '账户类型选错,流程结束', '在他行有久悬账户', '取消变更', '经营异常']
    100650 11325
    Index(['yd_applynote', 'yd_type', 'yd_bill_type'], dtype='object') 72996 46337
    79532
    258 12
    ['个人撤销', '预约信息有误', '重复提交', '营业执照影像未上传', '法人证件影像未上传', '营业执照影像文件未上传', '其他影像未上传', '客户取消开户', '公司其他原因', '上传影像不清晰', '重复预约', '资料不全', '客户取消销户', '客户尽调未通过', '其他影像资料未上传', '资料不齐', '法人证件影像文件未上传', '不符合账户设立标准', '客户取消', '账户性质错误', '资料不齐全', '客户资料不全', '账户性质有误', '开户许可证影像未上传', '无需预约', '账户性质错', '客户未来办理', '预约错误', '客户重复预约', '印章不符', '已开立账户', '客户取消办理', '非我行账户', '已销户', '重复', '核准类账户无需预约', '销户资料不全', '账户性质选择错误', '预约重复', '印鉴不符', '客户未到', '客户资料不齐全', '客户未办理', '取消销户', '其他', '客户资料不齐', '公章不符', '未办理', '差开户许可证影像', '户名有误', '资料有误', '客户未到网点办理', '账户性质选错', '客户预约错误', '账户类型错误', '核准类账户', '客户重复提交', '印章有误', '客户未到网点', '默认退票原因:标准电核105001电核不通过，详情：账户类型有误,流程结束', '其他原因', '另约时间', '核准类', '预约有误', '预约网点有误', '资料不完整', '开户许可证未上传', '企业不符合账户设立标准', '销户失败', '客户资料未带齐', '未销户', '客户改天来办理', '客户未至网点办理', '公司名称有误', '一般户未销户', '', '客户取消预约', '默认退票原因:标准电核105001电核不通过，详情：预约账户类型错误,流程结束', '有久悬账户', '非我网点账户', '重复申请', '非我行客户', '账户类型错', '默认退票原因:标准电核105001电核不通过，详情：账户类型选错,流程结束', '客户暂不销户', '名称有误', '信息有误', '户名不符', '退回', '1', '账户性质录入错误', '账户性质不符', '取消预约，影像未上传，16181', '账户类型选错', '客户未到场', '标准电核105001电核不通过，详情：,流程结束', '客户没来', '标准电核105001电核不通过，详情：客户预约账户类型有误,流程结束', '存在久悬账户', '暂不销户', '客户印章不符', '标准电核105001电核不通过，详情：预约账户类型错误,流程结束', '非本网点账户', '客户未来', '账户性质选择有误', '取消办理', '未上传影像', '未来办理', '默认退票原因:标准电核105001电核不通过，详情：账户类型选择错误,流程结束', '错', '客户未临柜', '基本户未变更', '重复提交申请', '不需要预约', '核准类无需预约', '业务未办理', '超时', '销户未成功', '标准电核105001电核不通过，详情：校验状态不通过，参数为空,流程结束', '销户资料准备不齐全', '其他退票原因:标准电核105001电核不通过，详情：客户预约账户类型有误,流程结束', '预约网点错误', '客户未至网点', '资料未带齐', '账户性质预约错误', '标准电核105001电核不通过，详情：账户类型选择错误,流程结束', '业务取消', '单位名称有误', '核准类账户，无需预约', '客户未销户', '一般户未注销', '已预约', '户名错误', '默认退票原因:标准电核105001电核不通过，详情：客户预约账户类型错误，重新预约,流程结束', '取消业务', '账户类型有误', '开户许可证遗失', '该账户已销户', '销户资料不齐', '取消预约，另约时间，16181', '营业执照过期', '预约信息错误', '客户提供资料不全', '请联系客户经理', '账户性质预约有误', '久悬', '重新预约', '销户资料不齐全', '账户名称有误', '未在我行开立账户', '有久悬', '账户类型不符', '核准类账户不用预约', '账户性质填写错误', '法人身份证过期', '公章有误', '未至办理', '预约时间有误', '非我社账户', '无影像资料', '名称错误', '客户印鉴不符', '未上传影像资料', '公章遗失', '重新预约办理时间', '未在我行开户', '账户类型选择错误', '客户改日办理', '网点预约错误', '录入错误', '客户公章有误', '预留印鉴不符', '核准类账户不需要预约', '标准电核105001电核不通过，详情：客户预约账户类型错误，重新预约,流程结束', '账户性质不对', '默认退票原因:标准电核105001电核不通过，详情：客户预约信息录入错误，重新预约,流程结束']
    1118 71854
    36
    79136
    ['个人撤销', '预约信息有误', '重复提交', '营业执照影像未上传', '法人证件影像未上传', '营业执照影像文件未上传', '其他影像未上传', '客户取消开户', '公司其他原因', '上传影像不清晰', '重复预约', '资料不全', '客户取消销户', '客户尽调未通过', '其他影像资料未上传', '资料不齐', '法人证件影像文件未上传', '客户取消', '不符合账户设立标准', '账户性质错误', '资料不齐全', '客户资料不全', '账户性质有误', '开户许可证影像未上传', '无需预约', '账户性质错', '客户未来办理', '预约错误', '客户重复预约', '印章不符', '客户取消办理', '已开立账户', '非我行账户', '已销户', '核准类账户无需预约', '重复', '销户资料不全', '账户性质选择错误', '预约重复', '客户未到', '印鉴不符', '客户资料不齐全', '客户未办理', '取消销户', '客户资料不齐', '其他', '客户预约账户类型有误,流程结束', '预约账户类型错误,流程结束', '公章不符', '账户类型有误,流程结束', '未办理', '差开户许可证影像', '资料有误', '户名有误', '客户未到网点办理', '账户性质选错', '客户预约错误', '账户类型选择错误,流程结束', '账户类型错误', '核准类账户', '客户重复提交', '印章有误', '客户未到网点', '账户类型选错,流程结束', '预约有误', '核准类', '另约时间', '其他原因', '资料不完整', '开户许可证未上传', '预约网点有误', '企业不符合账户设立标准']
    79136 9472
    452380
    '''

    import aigc
    import asyncio
    import nest_asyncio

    nest_asyncio.apply()

    aigc.AIGC_HOST = '47.110.156.41'
    cx = aigc.Local_Aigc(host=aigc.AIGC_HOST, use_sync=False, time_out=300)
    print(cx.chat(model='qwen:qwen3-32b', user_request='你好'))
    data = [{
        "name": "预约资料提交阶段",
        "children": [
            {
                "name": "信息填写问题",
                "children": [
                    {
                        "name": "企业名称未填写",
                        "semantic_templates": [
                            "客户在预约资料中未填写企业名称。",
                            "提交时未提供企业名称，资料不完整。",
                            "系统校验发现缺失企业名称字段。"
                        ]
                    },
                    {
                        "name": "法人姓名未填写",
                        "semantic_templates": [
                            "客户未填写法人姓名，无法继续预约。",
                            "法人信息缺失，缺少姓名字段。",
                            "提交内容中找不到法人姓名，校验失败。"
                        ]
                    },
                    {
                        "name": "身份证号格式错误",
                        "semantic_templates": [
                            "客户填写的身份证号码格式不正确。",
                            "身份证号不符合18位格式规范。",
                            "法人身份证信息填写格式异常。"
                        ]
                    },
                    {
                        "name": "账户类型未选择",
                        "semantic_templates": [
                            "客户未选择账户类型，预约信息不完整。",
                            "未指定账户类型（基本户/一般户等）。",
                            "系统校验提示账户类型未选择。"
                        ]
                    },
                    {
                        "name": "账户类型选择错误",
                        "semantic_templates": [
                            "客户实际需开基本户，误选为一般户。",
                            "客户办理的账户类型为外币账户开立，无需提前预约。",
                            "客户办理的账户类型为开设专用账户，无需提前预约，请客户直接到网点办理。",
                            "客户在开立账户时，误将专用账户选为基本账户。"
                        ]
                    },
                    {
                        "name": "开户目的描述为空或缺失",
                        "semantic_templates": [
                            "客户未填写开户用途，缺乏开户意图。",
                            "预约表单中开户目的未填写。",
                            "开户申请缺少账户用途说明。"
                        ]
                    }
                ]
            },
            {
                "name": "联系方式问题",
                "children": [
                    {
                        "name": "手机号未填写",
                        "semantic_templates": [
                            "客户预约时未填写联系电话。",
                            "缺少手机号，无法后续联系客户。",
                            "预约资料中未提供有效联系电话。"
                        ]
                    },
                    {
                        "name": "手机号格式错误",
                        "semantic_templates": [
                            "客户提交的手机号格式不正确。",
                            "联系电话格式异常，无法识别。",
                            "手机号字段填写格式不符合要求。"
                        ]
                    }
                ]
            },
            {
                "name": "影像资料提交问题",
                "children": [
                    {
                        "name": "未上传营业执照影像",
                        "semantic_templates": [
                            "客户未上传营业执照影像资料。",
                            "缺少营业执照图片，无法进行核验。",
                            "系统提示营业执照影像未提交。"
                        ]
                    },
                    {
                        "name": "未上传法人身份证影像",
                        "semantic_templates": [
                            "客户未提供法人身份证照片。",
                            "资料中缺失法人身份证影像。",
                            "未提交法人证件图片，无法继续。"
                        ]
                    },
                    {
                        "name": "未上传授权委托书（如适用）",
                        "semantic_templates": [
                            "代理人预约但未上传授权委托书。",
                            "缺失授权文件，无法确认委托关系。",
                            "授权人信息未提供，资料不合规。"
                        ]
                    },
                    {
                        "name": "上传字段错位（如身份证图传到营业执照字段）",
                        "semantic_templates": [
                            "上传资料字段填写错误，证件影像放错位置。",
                            "营业执照字段上传的是法人身份证图片。",
                            "上传资料字段对应错误，需重新提交。"
                        ]
                    },
                    {
                        "name": "文件上传失败或为空",
                        "semantic_templates": [
                            "上传的影像文件为空或失败。",
                            "客户上传文件缺失或损坏。",
                            "资料图像未能成功提交到系统中。"
                        ]
                    }
                ]
            },
            {
                "name": "格式与字段校验问题",
                "children": [
                    {
                        "name": "文件格式不支持（如 .exe, .zip）",
                        "semantic_templates": [
                            "客户上传了不支持的文件类型。",
                            "上传文件为压缩包或非法格式。",
                            "资料上传格式不符（如.exe/.zip）。"
                        ]
                    },
                    {
                        "name": "文件大小为0或超出限制",
                        "semantic_templates": [
                            "上传文件大小为0，疑似上传失败。",
                            "资料文件大小超出系统限制。",
                            "文件大小异常，无法识别。"
                        ]
                    },
                    {
                        "name": "未按要求填写或上传指定字段（如漏填开户类型）",
                        "semantic_templates": [
                            "客户未按表单要求完成全部信息。",
                            "资料中缺失关键字段，无法继续处理。",
                            "部分必填项未填写或未上传。"
                        ]
                    }
                ]
            },

            {
                "name": "预约行为问题",
                "children": [
                    {
                        "name": "重复提交预约",
                        "semantic_templates": [
                            "客户重复提交了相同的开户预约。",
                            "系统检测到多次相同预约信息。",
                            "存在重复预约记录，需合并处理。"
                        ]
                    },
                    {
                        "name": "预约时间段无效（如预约到非营业时间）",
                        "semantic_templates": [
                            "客户预约时间不在营业时段内。",
                            "预约时间与银行受理时间冲突。",
                            "选择了无效时间段进行预约。"
                        ]
                    },
                    {
                        "name": "选择网点与企业所在区域不匹配",
                        "semantic_templates": [
                            "客户选择的网点与企业地址不一致。",
                            "所选网点不属于企业所属区域。",
                            "客户跨区域预约开户，流程异常。"
                        ]
                    },
                    {
                        "name": "客户修改预约时间导致当前预约作废",
                        "semantic_templates": [

                            "客户修改预约时间，系统标记当前预约为失效。",
                            "因客户主动调整预约时间，本次预约状态中止。"
                        ]
                    },
                    {
                        "name": "选择业务类型错误（如选了对私业务）",
                        "semantic_templates": [
                            "客户误选为对私业务，实际为对公开户。",
                            "业务类型选择与预约目的不符。",
                            "客户实际为变更业务，误选为开户流程。",
                            "客户提交的银行账户信息已存在，实为变更申请，非开户业务。",
                            "客户已有账号，申请内容应为账户信息修改而非开户。",
                            "客户申请开立的账户类型不属于企业账户类别",

                            "客户选择的业务类型与实际办理的业务不符，应选择账户变更类型。"
                        ]
                    }

                ]
            }
        ]
    }
        , {
            "name": "材料预审阶段",
            "children": [
                {
                    "name": "资料缺失或无效",
                    "children": [
                        {
                            "name": "多项关键资料未提交",
                            "semantic_templates": [
                                "客户提交资料不完整，多个关键字段缺失。",
                                "营业执照、身份证及授权书均未上传，资料严重不足。",
                                "资料项普遍未提交，缺失严重，审核无法进行。"
                            ]
                        },
                        {
                            "name": "工商登记信息缺失或查询失败",
                            "semantic_templates": [
                                "系统未查到企业工商登记信息，无法进行身份核验。",
                                "工商平台无该企业记录，资料不完整。",
                                "企业信息在官方渠道中缺失，开户申请被退回。"
                            ]
                        },
                        {
                            "name": "营业执照影像缺失或无法识别",
                            "semantic_templates": [
                                "客户未提交营业执照影像或图像无法识别。",
                                "营业执照照片缺失或拍摄不清晰，无法确认信息。",
                                "系统无法识别营业执照图像内容，视为未提交。"
                            ]
                        },
                        {
                            "name": "法人身份证影像缺失或无法识别",
                            "semantic_templates": [
                                "法人身份证图像未上传或内容模糊。",
                                "未提交可用的法人身份证照片，信息无法核实。",
                                "法人证件图像无法识别关键字段，资料无效。"
                            ]
                        },
                        {
                            "name": "授权委托书缺失（如为代理人办理）",
                            "semantic_templates": [
                                "客户使用代理人办理但未提供授权委托书。",
                                "授权材料未上传，代理关系无法确认。",
                                "缺少代理授权文件，无法继续审核。"
                            ]
                        },
                        {
                            "name": "未提交实际经营场所证明材料（如租赁合同）",
                            "semantic_templates": [
                                "客户未提供实际办公地点证明材料。",
                                "缺失租赁合同或场所佐证文件，无法验证经营地址。",
                                "经营场所相关资料未提交，尽调受阻。"
                            ]
                        },
                        {
                            "name": "上传的是空白图、无文字内容",
                            "semantic_templates": [
                                "客户上传的图像为空白页，无法读取任何信息。",
                                "提交的文件无实际内容，视为无效材料。",
                                "资料图像中无有效文字或识别内容。"
                            ]
                        },
                        {
                            "name": "缺少辅助资料或证明文件",
                            "semantic_templates": [
                                "客户未提供辅助证件，无法完成资料审核。",
                                "开户需补充相关佐证材料，资料暂不合规。",
                                "缺少附加文件（如经营许可证、备案证明等），审核中止。"
                            ]
                        },
                        {
                            "name": "未刻制必要印章或签字样本",
                            "semantic_templates": [
                                "客户尚未完成公章和个人名章的刻制，开户资料不完整。",
                                "客户未能提供签字样本及印章印模，暂无法进行开户流程。",
                                "开户现场客户表示印章尚未刻制，资料准备不足。"
                            ]
                        }
                    ]
                },
                {
                    "name": "证件过期或无效",
                    "children": [
                        {
                            "name": "营业执照已过期",
                            "semantic_templates": [
                                "提交的营业执照已超出有效期，无法使用。",
                                "营业执照有效期已截止，不符合开户要求。",
                                "系统识别营业执照为过期状态。",
                                "企业的营业执照已被注销，不符合开户条件"
                            ]
                        },
                        {
                            "name": "法人身份证已过期",
                            "semantic_templates": [
                                "客户提交的身份证件已过期。",
                                "法人证件超过有效期，需更新后重新提交。",
                                "系统识别法人身份证为失效状态。"
                            ]
                        },
                        {
                            "name": "授权委托书过期或无效",
                            "semantic_templates": [
                                "授权书签署时间过早，已过有效期。",
                                "代理文件超出授权时限，需重新签署。",
                                "提交的委托书无效或不在有效期内。"
                            ]
                        },
                        {
                            "name": "提交的证件照片为复印件或电子截图",
                            "semantic_templates": [
                                "客户提交的是证件复印件或电子截图，非原件拍摄。",
                                "识别到上传图像非实拍照片，资料不符合要求。",
                                "仅提供复印版证件，需原件影像支持。"
                            ]
                        }
                    ]
                },
                {
                    "name": "资料伪造或真实性存疑",
                    "children": [
                        {
                            "name": "营业执照疑似伪造（二维码查验失败）",
                            "semantic_templates": [
                                "营业执照二维码查验失败，疑似伪造资料。",
                                "系统无法在工商平台查到对应营业执照。",
                                "提交营业执照疑似为虚假文件，信息无法验证。"
                            ]
                        },
                        {
                            "name": "法人身份证照片疑似篡改",
                            "semantic_templates": [
                                "身份证图像存在明显修改痕迹，疑似被篡改。",
                                "系统识别身份证内容与样式异常，涉嫌伪造。",
                                "法人证件图像内容不自然，存在可疑处理。"
                            ]
                        },
                        {
                            "name": "授权委托书签名不符或无法人签章",
                            "semantic_templates": [
                                "授权书上签名与法人信息不一致。",
                                "委托书无法人签章或签署页缺失。",
                                "代理授权文件签名无效，资料不合规。"
                            ]
                        },
                        {
                            "name": "提交的资料存在明显合成痕迹",
                            "semantic_templates": [
                                "资料图像有合成痕迹，涉嫌伪造或修改。",
                                "上传文件显示拼接、编辑或剪裁异常。",
                                "系统检测资料图像存在PS痕迹，需人工复核。"
                            ]
                        },
                        {
                            "name": "影像中出现与预约企业无关的信息（如他人资料）",
                            "semantic_templates": [
                                "影像资料显示的企业名称与预约企业不一致。",
                                "图像中展示的是他人资料，无法关联当前预约。",
                                "上传内容与本次预约主体无关。"
                            ]
                        }
                    ]
                },
                {
                    "name": "影像质量问题",
                    "children": [
                        {
                            "name": "照片模糊无法识别关键字段",
                            "semantic_templates": [
                                "图像模糊，无法识别证件号码或企业名称。",
                                "拍摄照片不清晰，关键信息无法提取。",
                                "资料照片模糊影响审核判断。"
                            ]
                        },
                        {
                            "name": "影像反光、遮挡、倾斜、裁切不完整",
                            "semantic_templates": [
                                "证件照片存在遮挡或反光，影响识别。",
                                "图像裁切不完整，信息被截断。",
                                "拍摄角度不正，导致无法清晰读取内容。"
                            ]
                        },
                        {
                            "name": "多页材料仅上传部分页（如营业执照首页）",
                            "semantic_templates": [
                                "客户仅上传资料首页，缺少完整页码。",
                                "多页文件未全部提交，资料不完整。",
                                "营业执照未提供副页，内容缺失。"
                            ]
                        },
                        {
                            "name": "上传图像为非原件翻拍（如电脑屏幕截图）",
                            "semantic_templates": [
                                "客户上传的是屏幕截图，非实物拍摄图片。",
                                "证件图像来源为电子翻拍，无法核验真实性。",
                                "非原件照片不符合资料提交要求。"
                            ]
                        }
                    ]
                },
                {
                    "name": "资料信息不一致",
                    "children": [
                        {
                            "name": "基本户核准号填写错误或无效",
                            "semantic_templates": [
                                "客户提供的基本户核准号格式错误或无效，系统校验失败。",
                                "录入的核准号不符合规范，无法进行一般户开户。",
                                "客户填写的基本户核准号与人行系统信息不匹配，审核未通过。"
                            ]
                        },
                        {
                            "name": "营业执照企业名称与预约信息不一致",
                            "semantic_templates": [
                                "营业执照上的企业名称与预约填写不符。",
                                "预约企业名称与资料中显示不一致。",
                                "提交材料中的单位名称与申请信息不一致。"
                            ]
                        },
                        {
                            "name": "法人身份证信息与预约登记不符",
                            "semantic_templates": [
                                "预约登记的法人姓名与身份证照片不一致。",
                                "证件信息与系统登记内容存在差异。",
                                "客户提交的身份证信息不匹配预约内容。",
                                "客户提供的法人信息与工商登记信息不符，需更新工商企业信息以确保法人信息一致。"
                            ]
                        },
                        {
                            "name": "营业执照地址与实际租赁合同地址不符",
                            "semantic_templates": [
                                "营业执照注册地址与租赁合同显示地址不一致。",
                                "企业注册地址与经营场所不符，存在疑点。",
                                "两份资料地址信息存在冲突。"
                            ]
                        },
                        {
                            "name": "授权书上法人姓名与身份证不一致",
                            "semantic_templates": [
                                "委托书签署人与身份证姓名不一致。",
                                "授权书中法人信息与证件内容不符。",
                                "代理人资料与法人授权信息无法对应。"
                            ]
                        },
                        {
                            "name": "统一社会信用代码在人行与工商系统中不一致",
                            "semantic_templates": [
                                "统一社会信用代码在人行系统与工商系统不一致，暂无法核实主体身份。",
                                "人行信息校验失败，代码与工商平台登记不一致。",
                                "统一代码系统校验异常，需客户核实后修正。"
                            ]
                        },
                        {
                            "name": "工商信息与人行基本户信息不一致",
                            "semantic_templates": [
                                "客户营业执照显示法人信息已变更，但人行系统未同步，开户无法进行。",
                                "注册地址在工商系统中已更新，但人行备案地址未变更，资料校验失败。",
                                "工商登记注册资本变更与人行存量信息不符，流程中止。",
                                "客户需先到基本存款账户开户行办理经营范围变更手续",
                                "工商登记注册资本与人行系统信息不一致，开户校验失败。",
                                "客户已进行监事人员变更，但开户系统信息未更新",
                                "客户经营范围变更后未到基本账户开户行进行相应变更手续"
                            ]
                        },
                        {
                            "name": "基本存款账户地区编码填写错误",
                            "semantic_templates": [
                                "客户填写的地区编码不符合标准格式，审核未通过。",
                                "地区编码与预约网点所在地区不一致，系统退回。",
                                "行政区划代码校验失败，需客户重新填写。"
                            ]
                        },
                        {
                            "name": "纳税信息与统一社会信用代码不一致",
                            "semantic_templates": [
                                "纳税人识别号与营业执照统一社会信用代码不一致，资料审核失败。",
                                "税务登记信息与工商登记主体不符，系统无法匹配企业身份。",
                                "税号校验失败，疑似为不同企业主体信息。"
                            ]
                        },
                        {
                            "name": "联系方式与工商登记信息不符",
                            "semantic_templates": [
                                "客户提供的法人手机号与工商系统登记号码不一致。",
                                "预约中登记的联系方式与官方注册信息不一致。",
                                "法人联系电话与工商备案信息无法对应，需进一步核实。"

                            ]
                        },
                        {
                            "name": "基本户开户许可证编号无效或不存在",
                            "semantic_templates": [
                                "客户提供的基本存款账户开户许可证编号在人行系统中未能查到。",
                                "系统核验发现提交的基本户编号无效或信息不一致。",
                                "开户许可证编号校验失败，视为无效资料。"
                            ]
                        },
                        {
                            "name": "登记信息联网核查失败",
                            "semantic_templates": [
                                "客户提供的登记信息未通过联网核查系统验证。",
                                "客户身份证联网核查失败，系统提示信息不存在。",
                                "姓名与身份证号码不一致，未通过公安系统校验。",
                                "客户提供身份证信息无效，未通过二要素验证。",
                                "资料中填写的信息无法在官方平台核实，存在差异。"
                            ]
                        }

                    ]
                },
                {
                    "name": "身份与联系方式核验失败",
                    "children": [
                        {
                            "name": "手机号未完成实名认证",
                            "semantic_templates": [
                                "负责人手机号未完成实名认证，系统校验失败。",
                                "客户提供的手机号码未通过实名验证，无法继续开户。",
                                "系统识别该手机号非实名登记，开户流程中止。"
                            ]
                        },
                        {
                            "name": "联系方式与身份信息不匹配",
                            "semantic_templates": [
                                "客户登记的手机号与法人身份不匹配。",
                                "预约中填写的联系电话无法确认为法人本人所有。",
                                "客户所留手机号为他人名下，未通过一致性校验。"
                            ]
                        }
                    ]
                },
                {
                    "name": "上传异常或系统问题",
                    "children": [
                        {
                            "name": "文件格式错误（如 .zip/.exe）",
                            "semantic_templates": [
                                "上传文件类型不符合要求，无法识别。",
                                "客户上传了压缩包或非法格式的资料。",
                                "系统提示上传文件格式错误。"
                            ]
                        },
                        {
                            "name": "上传失败或文件丢失",
                            "semantic_templates": [
                                "资料上传失败或未被系统接收。",
                                "提交的文件在系统中找不到记录。",
                                "客户操作异常，资料未成功保存。"
                            ]
                        },
                        {
                            "name": "多个文件内容冲突（如两份营业执照不同）",
                            "semantic_templates": [
                                "客户提交的多份资料内容互相矛盾。",
                                "上传了两份信息不一致的营业执照。",
                                "提交材料存在重复且冲突的信息。"
                            ]
                        },
                        {
                            "name": "字段错位上传（如法人证件上传到了营业执照字段）",
                            "semantic_templates": [
                                "客户上传材料字段对应错误。",
                                "身份证影像被误传到营业执照字段。",
                                "上传文件内容与字段不匹配，审核失败。"
                            ]
                        }
                    ]
                }
            ]
        }
        , {
            "name": "标准电核阶段",
            "children": [
                {
                    "name": "电话无法接通",
                    "children": [
                        {
                            "name": "客户未接听电话",
                            "semantic_templates": [
                                "客户未接听电话，未能完成电核。",
                                "多次拨打电话无人接听，沟通失败。",
                                "客户预约后电话一直处于无人接听状态。"
                            ]
                        },
                        {
                            "name": "预约电话为空或无效",
                            "semantic_templates": [
                                "客户填写的联系电话为空或格式异常。",
                                "提交的手机号为空值，无法拨打电话核实。",
                                "联系电话缺失或为无效号码。"
                            ]
                        },
                        {
                            "name": "号码错误或停机",
                            "semantic_templates": [
                                "拨打客户号码提示停机或不存在。",
                                "客户电话无法拨通，可能为错号。",
                                "电核过程中提示号码错误或未开通服务。",
                                "客户预约时预留的电话号码为空号，无法联系到客户"
                            ]
                        },
                        {
                            "name": "多次拨打均未接通",
                            "semantic_templates": [
                                "多次尝试联系客户，始终未接通电话。",
                                "电核期间多次拨打客户电话均无回应。",
                                "连续联系多次，客户始终未接听。"
                            ]
                        }
                    ]
                },
                {
                    "name": "身份与意愿核验失败",
                    "children": [
                        {
                            "name": "接听人非法人或无授权",
                            "semantic_templates": [
                                "接听电话者并非法人，也未能提供授权信息。",
                                "电话由他人接听，无法核实身份。",
                                "电话接听人不是法人或未持有效授权书。"
                            ]
                        },
                        {
                            "name": "客户拒绝确认身份信息",
                            "semantic_templates": [
                                "客户在电话中拒绝核实其身份信息。",
                                "对方不愿意提供身份验证信息，无法确认身份。",
                                "客户不愿配合身份确认流程。"
                            ]
                        },
                        {
                            "name": "客户表示无开户意愿",
                            "semantic_templates": [
                                "客户明确表示暂不考虑开户。",
                                "客户否认有开户需求或意愿。",
                                "对方表示没有申请开户，不需办理。"
                            ]
                        },
                        {
                            "name": "客户自称非预约人或不知情",
                            "semantic_templates": [
                                "客户在电话中表示自己未预约开户。",
                                "客户称不知情，也未授权他人预约开户。",
                                "客户否认提交过开户申请，存在疑点。"
                            ]
                        }
                    ]
                },
                {
                    "name": "开户用途不明确或异常",
                    "children": [
                        {
                            "name": "用途描述含糊（如“发展业务”但无具体内容）",
                            "semantic_templates": [
                                "客户仅表示用于‘发展业务’，未能详细说明。",
                                "开户用途描述过于笼统，缺乏具体内容。",
                                "客户无法解释‘业务发展’的具体场景或用途。"
                            ]
                        },
                        {
                            "name": "无法说明收入来源或交易对象",
                            "semantic_templates": [
                                "客户无法说明收入的来源或合作对象。",
                                "未能清晰阐述收付款的对象或业务背景。",
                                "资金往来的上游下游情况无法说明。"
                            ]
                        },
                        {
                            "name": "业务模式与开户类型不匹配",
                            "semantic_templates": [
                                "客户描述的业务与所选账户类型不符。",
                                "业务实际用途与申请的账户类别不匹配。",
                                "客户申请基本户，但用途偏向一般账户使用场景。"
                            ]
                        },
                        {
                            "name": "疑似用于违规目的（如代收代付、虚拟币）",
                            "semantic_templates": [
                                "客户用途描述涉及敏感或高风险用途（如代收）。",
                                "客户可能将账户用于虚拟币交易或转账。",
                                "电话中透露资金将用于非合规渠道。"
                            ]
                        }
                    ]
                },
                {
                    "name": "业务真实性存疑",
                    "children": [
                        {
                            "name": "客户对公司业务不熟悉",
                            "semantic_templates": [
                                "客户无法说明公司主营业务。",
                                "对业务内容不清楚，存在代办嫌疑。",
                                "客户对经营项目不了解，无法回应问题。"
                            ]
                        },
                        {
                            "name": "无法清晰说明经营内容/流程",
                            "semantic_templates": [
                                "客户无法具体说明业务运作流程。",
                                "经营模式模糊，无法支撑开户合理性。",
                                "客户答复空泛，缺乏实际经营细节。"
                            ]
                        },
                        {
                            "name": "客户语言模糊、前后矛盾",
                            "semantic_templates": [
                                "客户表达前后不一致，信息反复修改。",
                                "语言模糊、不连贯，缺乏可信度。",
                                "客户解释内容反复变更，存疑。"
                            ]
                        },
                        {
                            "name": "客户多次修改说辞或含糊其辞",
                            "semantic_templates": [
                                "客户在电话中多次更换说法。",
                                "客户无法给出明确解释，答非所问。",
                                "客户含糊其辞，未能清楚表达业务实情。"
                            ]
                        }
                    ]
                },
                {
                    "name": "沟通配合度差",
                    "children": [
                        {
                            "name": "客户因居住地偏远不便到行",
                            "semantic_templates": [
                                "客户因距离过远，难以亲自到网点办理业务。",
                                "客户居住偏远地区，出行不便导致办理中断。",
                                "客户反馈因交通限制难以按预约时间前往办理。"
                            ]
                        },
                        {
                            "name": "客户挂断电话或中断沟通",
                            "semantic_templates": [
                                "客户中途挂断电话，无法继续电核。",
                                "沟通过程中客户主动终止对话。",
                                "电话沟通过程中突然被挂断。"
                            ]
                        },
                        {
                            "name": "客户强烈抵触核查流程",
                            "semantic_templates": [
                                "客户对核查流程表示强烈不满。",
                                "拒绝配合电话访谈内容，态度激烈。",
                                "客户对开户核实表现出明显抵触情绪。"
                            ]
                        },
                        {
                            "name": "未按约定时间回拨",
                            "semantic_templates": [
                                "客户承诺回拨但未按时联系。",
                                "未在规定时间回复电话，造成流程中断。",
                                "电话核实约定时间未联系成功。"
                            ]
                        },
                        {
                            "name": "多次催促仍不回复",
                            "semantic_templates": [
                                "多次电话联系未获客户回应。",
                                "客户长期不回复信息，沟通中断。",
                                "反复尝试联系未果，配合度低。"
                            ]
                        }
                    ]
                },
                {
                    "name": "其他可疑行为",
                    "children": [
                        {
                            "name": "电话背景噪音异常（如多人提示）",
                            "semantic_templates": [
                                "电话背景嘈杂，有多人提示嫌疑。",
                                "电话中传出非自然对话声音，疑似操控。",
                                "通话过程中有他人干预指引客户应答。"
                            ]
                        },
                        {
                            "name": "同一号码关联多个预约（疑似中介批量）",
                            "semantic_templates": [
                                "系统检测到该手机号对应多个预约。",
                                "客户使用的号码与其他公司重复，疑似代办。",
                                "预约手机号涉及多企业，存在中介风险。"
                            ]
                        },
                        {
                            "name": "客户表示预约是他人代办（无授权）",
                            "semantic_templates": [
                                "客户称开户申请由他人操作，自己不清楚。",
                                "客户表示资料是他人代提交，未授权。",
                                "电话中承认他人操作，但无法提供授权证明。"
                            ]
                        }
                    ]
                }
            ]
        }
        , {
            "name": "尽职调查阶段",
            "children": [
                {
                    "name": "企业经营真实性存疑",
                    "children": [
                        {
                            "name": "实地走访无办公场所",
                            "semantic_templates": [
                                "实地走访发现企业地址为空，无办公迹象。",
                                "上门核查时未发现实际办公场所。",
                                "现场未见办公设备或人员，无法确认经营真实性。"
                            ]
                        },
                        {
                            "name": "注册地址为虚拟地址（如挂靠、孵化器）",
                            "semantic_templates": [
                                "企业注册地址为孵化器，无法确认是否真实办公。",
                                "公司登记地址为虚拟场所，实际无人员办公。",
                                "注册地址显示为共享办公或挂靠地址。"
                            ]
                        },
                        {
                            "name": "实际经营地与营业执照地址严重不符",
                            "semantic_templates": [
                                "公司实际经营地点与营业执照登记地址不一致。",
                                "企业提供的经营场所与注册信息冲突。",
                                "现场地址与工商登记位置严重偏离。"
                            ]
                        },
                        {
                            "name": "办公场所无人办公或无人知晓企业存在",
                            "semantic_templates": [
                                "办公地点人员表示不知该企业存在。",
                                "走访时现场无人办公，疑似空壳。",
                                "场所租户与企业无关，核查失败。"
                            ]
                        },
                        {
                            "name": "公司无固定电话、无网站、无线上痕迹",
                            "semantic_templates": [
                                "公司无法提供固定电话或网站信息。",
                                "缺乏线上资料，无法核实企业存在。",
                                "企业未留联系方式，无网络可查记录。"
                            ]
                        }
                    ]
                },
                {
                    "name": "尽调发现不符合开户条件",
                    "children": [
                        {
                            "name": "实际经营场所尚未准备或不符要求",
                            "semantic_templates": [
                                "客户营业场所尚未准备完毕，不符合开户条件。",

                                "客户提交的注册地址尚未启用，暂不予开户。"
                            ]
                        }
                    ]
                },
                {
                    "name": "企业业务活动异常",
                    "children": [
                        {
                            "name": "无法提供交易合同或上下游信息",
                            "semantic_templates": [
                                "客户无法出示交易合同或合作方清单。",
                                "未能提供上下游客户或供应商信息。",
                                "缺乏业务往来的证明材料。"
                            ]
                        },
                        {
                            "name": "客户无法说明收入来源或客户群体",
                            "semantic_templates": [
                                "客户无法说明主要收入来源。",
                                "未能明确企业客户群体及业务流向。",
                                "对资金来源未能做出有效解释。"
                            ]
                        },
                        {
                            "name": "对业务模式解释不清、无具体案例",
                            "semantic_templates": [
                                "客户对业务流程描述模糊，无实际案例支撑。",
                                "无法举出具体项目说明其经营内容。",
                                "业务逻辑不清晰，答复空泛。"
                            ]
                        },
                        {
                            "name": "主营业务与工商登记严重不符",
                            "semantic_templates": [
                                "客户主营业务与营业执照登记范围严重不符。",
                                "实际经营方向与注册信息差异较大。",
                                "工商登记显示为咨询，实际涉及资金交易。"
                            ]
                        }
                    ]
                },
                {
                    "name": "企业风险信息命中",
                    "children": [
                        {
                            "name": "命中法院被执行人/失信名单",
                            "semantic_templates": [
                                "企业或法人命中失信名单，存在执行记录。",
                                "系统查询发现客户为被执行人。",
                                "客户主体被列入法院限制名单。",
                                "客户为失信被执行人，命中法院限制高消费名单。",
                                "系统识别法人存在失信记录，无法继续开户。",
                                "客户主体在法院执行系统中被列为限制高消费对象。"
                            ]
                        },
                        {
                            "name": "命中洗钱高风险行业名单",
                            "semantic_templates": [
                                "企业行业类型属于洗钱高风险行业库。",
                                "经营范围命中高风险行业清单。",
                                "客户所在行业涉洗钱高风险警示系统。"
                            ]
                        },
                        {
                            "name": "法人或企业存在较多司法诉讼记录",
                            "semantic_templates": [
                                "系统查询发现客户存在大量诉讼记录。",
                                "企业近年频繁涉入司法案件。",
                                "客户涉及法律纠纷，存在较大风险。"
                            ]
                        },
                        {
                            "name": "企业为新注册公司且注册资本异常（如1亿实缴0）",
                            "semantic_templates": [
                                "公司刚成立不久，注册资本远超行业水平。",
                                "新注册企业资本结构异常（实缴为0）。",
                                "注册资金过高但缺乏实际经营支撑。"
                            ]
                        },
                        {
                            "name": "客户存在久悬账户记录",
                            "semantic_templates": [
                                "客户在他行存在久悬账户，当前申请需人工复核。",
                                "系统识别客户名下存在长期不动户记录，开户受限。",
                                "客户因久悬账户风险未通过开户审批。"
                            ]
                        }
                    ]
                },
                {
                    "name": "中介代办或异常操作痕迹",
                    "children": [
                        {
                            "name": "中介批量代办多个企业（同IP/手机号）",
                            "semantic_templates": [
                                "系统识别客户信息与多个企业重合，疑似中介代办。",
                                "同一手机号关联多家企业提交开户申请。",
                                "客户所用IP地址为中介高频操作源。"
                            ]
                        },
                        {
                            "name": "法人本人对开户流程不了解、明显被操控",
                            "semantic_templates": [
                                "法人对开户流程不清楚，疑似他人指导应答。",
                                "客户答复内容明显为他人代答或操控。",
                                "法人本人无法独立回答业务相关问题。"
                            ]
                        },
                        {
                            "name": "多个企业共用同一经营场所或联系人",
                            "semantic_templates": [
                                "发现多家公司共用同一联系人或办公地址。",
                                "客户注册地址与其他企业完全一致。",
                                "多家企业共享同一经营资料，存在代办嫌疑。"
                            ]
                        },
                        {
                            "name": "客户使用代理人常用术语（如“你们怎么审核”）",
                            "semantic_templates": [
                                "客户在沟通中频繁使用代理话术。",
                                "言谈内容与常见代办一致，回答模板化。",
                                "客户表现出非本人操作迹象。"
                            ]
                        }
                    ]
                },
                {
                    "name": "合规调查未通过",
                    "children": [
                        {
                            "name": "营业执照未年检或年报异常",
                            "semantic_templates": [
                                "企业营业执照未完成年检，流程中止。",
                                "工商系统显示企业年报异常，暂缓开户。",
                                "客户主体因年检未完成被列为经营风险企业。"
                            ]
                        },
                        {
                            "name": "资料与国家禁止行业有关（如地下钱庄、虚拟币）",
                            "semantic_templates": [
                                "客户经营内容涉及国家禁止行业（如虚拟币）。",
                                "资料显示与非法金融活动相关，合规不通过。",
                                "客户用途说明涉及地下钱庄、博彩等行业。"
                            ]
                        },
                        {
                            "name": "业务模式涉及代收代付、对私结算等高风险结构",
                            "semantic_templates": [
                                "客户用途为代收代付，存在资金风险。",
                                "开户目的涉及对私结算，不符合监管要求。",
                                "客户意图使用账户进行资金通道操作。"
                            ]
                        },
                        {
                            "name": "客户拒绝配合进一步尽调说明或拒绝补充材料",
                            "semantic_templates": [
                                "客户对进一步尽调要求不予配合。",
                                "拒绝提供补充资料，调查中断。",
                                "多次催促后仍未补全尽调材料，流程终止。"
                            ]
                        }
                    ]
                },
                {
                    "name": "尽调未完成",
                    "children": [
                        {
                            "name": "客户未按约定提供尽调补充材料",
                            "semantic_templates": [
                                "客户未在约定时间提交补充材料。",
                                "尽调所需资料未能按时提供，流程搁置。",
                                "客户材料提交超时，尽调无法继续。"
                            ]
                        },
                        {
                            "name": "尽调任务超时未反馈，流程中止",
                            "semantic_templates": [
                                "尽调任务已超出办理期限，系统关闭流程。",
                                "调查流程长时间未更新，自动终止。",
                                "尽调节点长时间未处理，标记为未完成。"
                            ]
                        },
                        {
                            "name": "客户联系中断，无法继续核查",
                            "semantic_templates": [
                                "客户无法联系，尽调工作中断。",
                                "电话、微信均无法取得联系，调查停滞。",
                                "客户长时间未回复，尽调无法推进。"
                            ]
                        }
                    ]
                },
                {
                    "name": "企业信用与合规状态异常",
                    "children": [
                        {
                            "name": "企业在工商系统中显示经营异常",
                            "semantic_templates": [
                                "企业被工商系统列入经营异常名录，开户暂缓。",
                                "工商登记状态异常，提示企业处于异常经营状态。",
                                "系统识别企业为异常户，流程无法继续。",
                                "企业被列入异常经营名录"
                            ]
                        },
                        {
                            "name": "企业被列入严重违法失信名单",
                            "semantic_templates": [
                                "企业列入严重违法名单，开户申请驳回。",
                                "工商系统显示企业为严重违法主体。",
                                "客户主体存在严重失信记录，开户被拒。"
                            ]
                        }
                    ]
                }

            ]
        }
        , {
            "name": "法人面签与双录阶段",
            "children": [
                {
                    "name": "面签人员非法人本人",
                    "semantic_templates": [
                        "前来面签的人员非法人本人，身份不符。",
                        "现场办理人员与营业执照法人不一致。",
                        "客户派其他人员代替法人到场，未授权。"
                    ]
                },
                {
                    "name": "法人未携带有效证件",
                    "semantic_templates": [
                        "法人未携带身份证，无法完成身份确认。",
                        "缺少原件证件，不能现场完成面签验证。",
                        "面签时无法出示有效身份证明材料。"
                    ]
                },
                {
                    "name": "法人拒绝配合双录",
                    "semantic_templates": [
                        "客户拒绝视频录音录像，双录环节无法完成。",
                        "法人明确表示不接受音视频双录。",
                        "因客户不配合，双录流程中断。"
                    ]
                },
                {
                    "name": "双录音视频不合规",
                    "semantic_templates": [
                        "双录音视频存在遮挡或录制异常。",
                        "录音录像设备未完整记录法人答复。",
                        "视频模糊或音频缺失，无法满足监管要求。"
                    ]
                },
                {
                    "name": "法人表达与前期信息不一致",
                    "semantic_templates": [
                        "法人在面签时的表述与电核内容不一致。",
                        "面签过程中法人用途说明与系统登记不符。",
                        "法人解释与前期电话访谈存在冲突。"
                    ]
                },
                {
                    "name": "法人行为异常或神情异常",
                    "semantic_templates": [
                        "面签过程中法人行为反常，无法正常沟通。",
                        "客户面签时神情紧张、不愿作答。",
                        "法人表现异常，疑似受他人操控或非自愿开户。"
                    ]
                },
                {
                    "name": "双录内容缺失或中断",
                    "semantic_templates": [
                        "双录未完成录制流程，系统中无完整记录。",
                        "双录录像内容中断，无法回放全流程视频。",
                        "音视频录制中途停止，资料不完整。"
                    ]
                },
                {
                    "name": "未按规定流程宣读风险提示",
                    "semantic_templates": [
                        "办理人员未对法人进行完整的风险提示宣读。",
                        "双录过程未包含完整的开户合规提醒环节。",
                        "风险提示未录入音视频，流程无效。"
                    ]
                }
            ]
        }
        , {
            "name": "账户开立阶段",
            "children": [
                {
                    "name": "身份认证失败",
                    "children": [
                        {
                            "name": "法人人脸识别未通过",
                            "semantic_templates": [
                                "开户过程中法人面部识别未通过，无法继续开户流程。",
                                "系统人脸识别校验失败，身份认证未通过。",
                                "法人身份未通过刷脸认证，开户中止。"
                            ]
                        }
                    ]
                },
                {
                    "name": "账户唯一性规则限制",
                    "semantic_templates": [
                        "客户已在他行开立基本账户，无法再次开立。",
                        "系统校验发现客户已有基本户，当前申请中止。",
                        "企业基本户已存在，开户申请不符合监管要求。"
                    ]
                },
                {
                    "name": "开户前需完成账户信息变更",
                    "semantic_templates": [
                        "客户需先在基本存款账户开户行完成变更后方可开户。",
                        "客户现有账户信息未变更，无法进行新账户申请。",
                        "当前开户申请前需完成原账户信息的变更手续。",
                        "因客户原账户信息未更新，开户预约暂缓。"
                    ]
                },
                {
                    "name": "存在未结清的账户业务冲突",
                    "semantic_templates": [
                        "企业存在未完成的基本户相关业务，暂无法开立新账户。",
                        "客户基本户业务未结清，当前开户申请中止。",
                        "检测到已有未完结的账户流程，影响本次受理。"

                    ]
                },
                {
                    "name": "开户审批流程未启动或未完成",
                    "semantic_templates": [
                        "账户申请处于待核准状态，审批流程尚未完成。",
                        "开户流程卡在审批环节，待审批。",
                        "系统显示账户未被核准，流程中断。"
                    ]
                },
                {
                    "name": "开户信息存在重大疑点",
                    "semantic_templates": [
                        "开户资料中存在重大疑点，开户中止。",
                        "系统识别客户信息异常，需进一步核实。",
                        "客户背景或信息存在风险，暂停开立账户。"
                    ]
                },
                {
                    "name": "开户条件不符合监管规定",
                    "children": [
                        {
                            "name": "客户基本户未完成注资变更手续",
                            "semantic_templates": [
                                "客户基本存款账户未完成注资变更，开户无法继续。",
                                "人行系统显示基本户尚未注资，开户申请驳回。",
                                "未完成资金实缴流程，暂不符合开立其他账户条件。"
                            ]
                        }
                    ],
                    "semantic_templates": [
                        "客户未能满足账户开立的监管条件。",
                        "资料和用途不符合监管账户分类要求。",
                        "不满足《账户管理办法》中的合规条件。"
                    ]
                },
                {
                    "name": "开户系统异常或流程中断",
                    "semantic_templates": [
                        "开户系统出现故障，流程未能完成。",
                        "开户环节中断，系统未成功生成账号。",
                        "因技术问题，开户申请未被处理完成。"
                    ]
                },
                {
                    "name": "风控或合规系统阻断开户",
                    "semantic_templates": [
                        "开户过程被风控系统自动阻断。",
                        "合规系统提示风险，开户流程终止。",
                        "客户被识别为高风险主体，开户被拒。"
                    ]
                },
                {
                    "name": "开户成功但未激活账户",
                    "semantic_templates": [
                        "账户已开立但客户未办理激活流程。",
                        "账户开立成功但未完成启用。",
                        "客户未到现场或未提交补充材料，账户处于未激活状态。"
                    ]
                },
                {
                    "name": "开户已完成，后续资料补交失败",
                    "semantic_templates": [
                        "客户开户后未能按时补交缺失资料。",
                        "账户开立完成，但因资料问题未能正式启用。",
                        "后续资料不符合要求，账户被暂停。"
                    ]
                }
            ]
        }
        , {
            "name": "行为驱动类中断原因",
            "children": [
                {
                    "name": "客户主动放弃类原因",
                    "children": [
                        {
                            "name": "客户决定不开户",
                            "semantic_templates": [
                                "客户表示无需开户，主动终止申请流程。",
                                "客户因自身业务调整决定不开户。",
                                "客户放弃开户申请，系统已作废当前流程。",
                                "客户暂不前来办理业务"
                            ]
                        },
                        {
                            "name": "客户改为他行开户",
                            "semantic_templates": [
                                "客户选择他行开户，取消我行开户申请。",
                                "客户说明已在其他银行完成开户，不再继续。",
                                "客户表示更倾向他行服务，取消当前预约。"
                            ]
                        },
                        {
                            "name": "客户因流程等待时间过长放弃",
                            "semantic_templates": [
                                "客户反馈流程太慢，放弃开户。",
                                "客户沟通多次未果，决定终止开户申请。",
                                "客户失去耐心选择退出当前开户流程。"
                            ]
                        }
                    ]
                },
                {
                    "name": "预约行为中断",
                    "children": [
                        {
                            "name": "客户更改预约时间，流程暂停待重新预约",
                            "semantic_templates": [
                                "客户主动更改预约时间，系统中止当前流程，待客户重新提交。",
                                "因客户时间调整，流程暂缓，等待新预约发起。",
                                "客户表示近期无法到访，原预约取消，将重新选择时间办理。",
                                "客户因临时有事，表示改天再来办理业务",
                                "法人无法到达，需重新安排时间"
                            ]
                        },
                        {
                            "name": "客户更换我行其他网点预约开户",
                            "semantic_templates": [
                                "客户表示原网点不方便，已改至我行其他网点重新预约。",
                                "客户更换开户网点，当前预约作废，已在本行其他网点预约成功。",
                                "客户因距离或交通问题改约本行其他支行办理开户。",
                                "客户放弃当前网点预约，已提交我行另一网点的新预约。"
                            ]
                        },
                        {
                            "name": "客户修改预约信息导致当前流程作废",
                            "semantic_templates": [
                                "客户重新预约，原预约流程自动关闭。",
                                "因客户主动调整预约信息，本次预约状态中止。"

                            ]
                        },
                        {
                            "name": "客户未到访网点办理预约业务",
                            "semantic_templates": [
                                "客户预约后未按时到网点进行办理，系统按期关闭流程。",
                                "预约超时未办理，系统自动关闭流程。"

                            ]
                        }
                    ]
                },
                {
                    "name": "客户未配合流程推进",
                    "children": [
                        {
                            "name": "客户长时间未回复信息或不接听电话",
                            "semantic_templates": [
                                "多次联系客户无果，流程自动中止。",
                                "客户电话长期无人接听，流程终止。",
                                "客户未反馈必要材料，系统中止流程。"
                            ]
                        },
                        {
                            "name": "客户拒绝补充材料或参与后续环节",
                            "semantic_templates": [
                                "客户拒绝补交授权委托书，无法继续开户。",
                                "客户不愿参加面签，主动放弃流程。",
                                "客户明确表示不配合尽调，不再继续流程。"
                            ]
                        }
                    ]
                }
            ]
        }
        , {
            "name": "银行内部操作异常",
            "children": [
                {
                    "name": "客户经理未提交开户申请",
                    "semantic_templates": [
                        "客户经理未在系统发起开户流程，预约流程中止。",
                        "资料审核通过后，未见开户申请提交记录。",
                        "因客户经理未完成开户系统操作，预约失效。"
                    ]
                },
                {
                    "name": "开户审批流程未启动",
                    "semantic_templates": [
                        "审核流程未被触发，开户申请滞留。",
                        "系统未流转至审批节点，流程中断。",
                        "客户申请未被提交至后台审批，流程终止。"
                    ]
                },
                {
                    "name": "审批长时间未处理",
                    "semantic_templates": [
                        "开户申请已提交，审批环节长时间未响应。",
                        "客户资料挂起，审批迟迟未完成。",
                        "由于审批流程积压，开户超时未完成。"
                    ]
                },
                {
                    "name": "内部审核状态异常",
                    "semantic_templates": [
                        "系统状态显示资料已审核，但流程未继续。",
                        "审核状态标记与实际流程进度不一致。",
                        "客户状态卡在审核节点，未进入后续流程。"
                    ]
                },
                {
                    "name": "系统提交失败或中断",
                    "semantic_templates": [
                        "开户申请系统提交时异常中断，流程未发起。",
                        "系统提交开户信息失败，状态未更新。",
                        "提交后系统异常，流程未被记录。"
                    ]
                },
                {
                    "name": "预约排期冲突或资源不足",
                    "semantic_templates": [
                        "我行开户预约已排满，建议客户改至他行或延期办理。",
                        "预约排期过长，客户难以等待，转向其他渠道开户。",
                        "当前预约日期已排至月底，客户申请受限。"
                    ]
                },
                {
                    "name": "网点未完成上门核实",
                    "semantic_templates": [
                        "标准电核已完成，但支行/网点未完成上门核查，流程自动中止。",
                        "因业务人员未上门核实客户资料，导致开户流程结束。",
                        "电核通过但未履行现场确认义务，开户被退回处理。"
                    ]
                }
            ]
        }
    ]
    # df_template, json_structure = make_template(data)
    df_template = pd.read_excel('data/semantic_templates.xlsx')
    json_structure = load_dictjson('data/semantic_templates_structure.json')
    print(len(json_structure), len(f'{json_structure}'))

    sample_df = pd.read_excel('data/sample_开户过程分类优化8.1.xlsx')
    sample_df_2 = pd.read_excel('data/sample_开户过程分类优化8.3.xlsx')
    multi_cleaned = pd.read_excel('data/combined_multi_cleaned_kmeans.xlsx', index_col=0)  # multi_cleaned_kmeans.xlsx
    mask = (~multi_cleaned.原始句子.isin(sample_df.原始句子.unique())) & (
        ~multi_cleaned.原始句子.isin(sample_df_2.原始句子.unique())) & (multi_cleaned.原始句子.str.len() >= 10)
    print(mask.sum())
    df_sample = multi_cleaned[mask].groupby('final_cluster_id', group_keys=False).apply(
        lambda x: x.sample(n=min(2, len(x)), random_state=42)).sample(n=200, random_state=42).reset_index(drop=True)
    print(df_sample.shape)
    merged_df = asyncio.run(
        structure_sample_applynote(cx, df_sample, df_template, json_structure, model='qwen:qwen3-32b',
                                   save_name='data/sample_开户过程分类优化8.5.xlsx'))

    print(merged_df.head())
    cx.close()
