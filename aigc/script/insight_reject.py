import numpy as np
import pandas as pd
import os, re
import psutil
from generates import call_ollama
from utils import extract_json_from_string

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
    load_applynote_script()
    '''
    示例结果

    '''
