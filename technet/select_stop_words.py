import json, os, re
import random
import numpy as np
import pandas as pd


# import datetime
# os.path.join(DATA_DIR,data/


def call_select_words(word_data, limit=4):
    data = word_data[(word_data['w2v'] == 1) & (word_data['停用标记'] == 0)]
    if data.shape[0] < limit:
        print('完成停用标记')
        return
    number = random.randint(0, data.shape[0] - limit)
    # print(number,data.shape[0])
    select_words = data.iloc[number:number + limit]
    return select_words.index.to_list(), select_words.to_json(force_ascii=False)  # orient='records',


def set_stop_flag(word_data, words, stops, uid=0):  # stop>0
    flags = [((1 if x else -1) << uid) for x in stops]
    mask = word_data.index.isin(words)
    stop_flag = (word_data.loc[mask, '停用标记'].values | flags)
    word_data.loc[mask, '停用标记'] = stop_flag
    return stop_flag


class StopWordsFlag:
    user_data = []  # [{:,:[(,),]}]#"name": "abc", "randoms": 1, "cross": 0, "readn": 91, "stopn": 12, "task":[]

    def __init__(self):
        self.word_data = None
        self.stop_words = []

    def load(self, word_data):
        if not word_data.empty:
            self.word_data = word_data[(word_data['w2v'] == 1)].rename(
                columns={'cut_max_count': 'cmc', 'doc_max_count': 'dmc', 'doc_max_index': 'dmc_idx'}).sort_values(
                ['dmc_idx', 'dmc', 'cmc', 'tf'], ascending=[True, False, False, False]).copy()

            self.word_data['阅读标记'] = 0
            self.word_data['停用标记'] = 0

        print('数据导入:', self.word_data.columns, self.word_data.shape)

    def load_file(self):
        try:
            path = 'data/patent_word_stop_flag.xlsx'
            if os.path.exists(path):
                word_data_stop = pd.read_excel(path, index_col=0)
                if not word_data_stop.empty:
                    del self.word_data['阅读标记']
                    del self.word_data['停用标记']
                    self.word_data = pd.merge(self.word_data, word_data_stop, left_index=True, right_index=True,
                                              how='left')
                    self.get_stop_words(-1)

            path = 'data/user_data.josn'
            if os.path.exists(path):
                with open(path, 'r') as file:
                    self.user_data = json.load(file)
        except KeyError:
            print(f"KeyError: {self.word_data}")
        except Exception as e:
            print(f"Error occurred load_file: {str(e)}")
        finally:
            pass

    def save_file(self):
        self.word_data[['停用标记', '阅读标记']].to_excel('data/patent_word_stop_flag.xlsx')
        with open('data/user_data.josn', 'w') as file:
            json.dump(self.user_data, file)

    def save_data(self):
        with open('data/word_stop_flag.josn', 'w') as file:  # datetime.datetime.now().date()
            json.dump(self.word_data.loc[self.word_data['停用标记'] != 0, '停用标记'].to_dict(), file)
        pd.DataFrame(self.user_data).to_excel('data/user_data.xlsx')

    def flag_table(self, mask=[]):
        table = self.word_data.loc[mask if len(mask) else (self.word_data['阅读标记'] != 0),
                                   ['阅读标记', '停用标记']].reset_index()
        table.columns = ['word', 'read_flag', 'stop_flag']
        return table

    def reset(self):
        self.word_data['阅读标记'] = 0
        self.word_data['停用标记'] = 0
        self.user_data = []
        self.stop_words = []

    def register_user(self, name, **kwargs):
        names = [v.get('name') for v in self.user_data]
        if name in names:
            uid = names.index(name)
            return uid, self.user_data[uid]  # self.user_data[uid].get('uid',uid)
        elif len(names) < 128:
            uid = len(self.user_data)
            inf = kwargs if len(kwargs) else {'uid': uid, 'name': name,
                                              'randoms': 1, 'cross': 0, 'readn': 0, 'stopn': 0, 'task': []}
            self.user_data.append(inf)
            return uid, inf

        return -1, dict()

    def get_user(self, name):
        for i, x in enumerate(self.user_data):
            if x.get('name') == name:
                return x.get('uid', i), x
        return -1, dict()

    def set_user(self, uid=0, **kwargs):
        if uid >= 0 and uid < len(self.user_data):
            self.user_data[uid].update(kwargs)
            return self.user_data[uid]

    def reset_user_data(self):
        for uid in range(len(self.user_data)):
            self.user_data[uid]['readn'] = sum((self.word_data['阅读标记'] & (1 << uid)) != 0)
            self.user_data[uid]['stopn'] = sum((self.word_data['停用标记'] & (1 << uid)) != 0)

    def search_words(self, words):
        tokens = re.split(r'[^\w\s]| ', words)
        tokens = [token.strip().lower() for token in tokens]
        return self.word_data[self.word_data.index.isin(tokens)]

    def get_stop_words(self, uid=-1):
        if uid < 0:
            mask = self.word_data['停用标记'] != 0
            self.stop_words = self.word_data[mask].index.to_list()
        else:
            mask = (self.word_data['停用标记'] & (1 << uid)) != 0

        return self.word_data[mask].index

    def set_words_flag(self, words, uid, setstop):
        tokens = re.split(r'[^\w\s]| ', words)
        tokens = [token.strip().lower() for token in tokens]
        mask = self.word_data.index.isin(tokens)
        if mask.sum() == 0:
            return [], tokens
        if setstop:  # 置位
            self.word_data.loc[mask, '停用标记'] |= np.array([1 << uid] * mask.sum())
            # self.stop_words += [w for w in self.word_data[mask].index if w not in self.stop_words]
        else:  # 清零
            if uid >= 0:
                self.word_data.loc[mask, '停用标记'] &= np.array([~(1 << uid)] * mask.sum())
                # self.stop_words=[w for w in self.stop_words if w not in self.word_data[mask].index]
            else:  # 管理员清零
                self.word_data.loc[mask, '停用标记'] = 0

        return mask, self.word_data[mask].index.to_list()

    def call_back_words(self, uid=0):
        task_words = self.user_data[uid].get('task', [])
        words = [w for task in task_words for w in task]
        data = self.word_data[self.word_data.index.isin(words)]
        if data.shape[0] == 0:
            return [], pd.DataFrame()

        words_table = data.copy()
        mask = words_table['dmc_idx'].duplicated(keep='first')
        words_table.loc[mask, ['标题 (中文)', '摘要 (中文)']] = '同上'

        return words_table.index.to_list(), words_table

    def call_select_words(self, limit=4, randoms=True, cross=False, uid=0):
        if uid >= 0 and cross:
            data = self.word_data[(self.word_data['阅读标记'] & (1 << uid)) == 0]
        else:  # 筛选去除其他用户重复已读
            data = self.word_data[self.word_data['阅读标记'] == 0]  # &(self.word_data['停用标记'] == 0)

        if data.shape[0] < limit:
            print('完成停用标记')
            return [], pd.DataFrame()

        start = random.randint(0, data.shape[0] - limit) if randoms else 0
        words_table = data.iloc[start:start + limit].copy()
        words_select = words_table.index.to_list()

        mask = words_table['dmc_idx'].duplicated(keep='first')
        words_table.loc[mask, ['标题 (中文)', '摘要 (中文)']] = '同上'

        if uid >= 0:
            if not self.user_data[uid].get('task'):
                self.user_data[uid]['task'] = []
            self.user_data[uid]['task'].append(tuple(words_select))
            if len(self.user_data[uid]['task']) > 5:
                self.user_data[uid]['task'].pop(0)

        return words_select, words_table  # to_json(force_ascii=False)

    def set_stop_flag(self, stops: list, uid=0):  # stop>0
        task_words = self.user_data[uid].get('task', [])
        words = task_words[-1]
        last = 1
        if len(stops) > len(words) or len(set(stops) - set(words)) > 0:
            words = [w for task in task_words for w in task]
            last = 0
        mask = self.word_data.index.isin(words)
        if mask.sum() == 0:
            print(f'数据错误：{words},{stops}')
            return [], []

        words = self.word_data[mask].index
        flags = [((1 << uid) if w in stops else 0) for w in words]
        self.word_data.loc[mask, '停用标记'] = (self.word_data.loc[mask, '停用标记'].values | flags)  # stop_flag
        self.word_data.loc[mask, '阅读标记'] |= np.array([1 << uid] * len(words))
        self.stop_words += [w for w in stops if w not in self.stop_words]
        if last:
            self.user_data[uid]['readn'] += len(words)
            self.user_data[uid]['stopn'] += len(stops)
        # print(stops, flags, self.user_data[uid], '\n', self.word_data.loc[mask, ['停用标记', '阅读标记']])
        return mask, flags  # words


if __name__ == "__main__":
    swg = StopWordsFlag()
    word_data = pd.read_excel('data/patent_doc_cut_word.xlsx', index_col=0, nrows=1000)
    swg.load(word_data)
    swg.reset()
    swg.register_user('test')

    pass
