from collections import deque
import itertools
from rime.allele import Allele


def stream_print_genotypes(gen_iter, max_batches=9, start_batch=8):
    """
    Stream printing:
      - 第1批：8 个（每行 2 个，4 行）
      - 第2批：16 个（每行 4 个，4 行）
      - 第3批：32 个（每行 8 个，4 行）
      ...
    这里的阈值是累计阈值（cumulative），确保每次打印的是“下一批的增量”。
    """
    buffer = deque()
    # 生成增量批次大小：8,16,32,...
    batch_sizes = [start_batch * (i + 1) for i in range(max_batches)]
    # 生成累计阈值：8, 8+16=24, 24+32=56, ...
    thresholds = []
    cum = 0
    for sz in batch_sizes:
        cum += sz
        thresholds.append(cum)

    current_idx = 0
    for count, item in enumerate(gen_iter):
        buffer.append(item)

        # 触发所有已经达到的累计阈值（一般只会触发一次）
        while current_idx < len(thresholds) and count >= thresholds[current_idx]:
            # 这个批次要打印的条目数（增量大小）
            batch_size = batch_sizes[current_idx]  # 注意：用增量大小作为本批打印量
            cols = max(1, batch_size // 4)  # 每批 4 行
            print(f"\n=== 第 {current_idx + 1} 批 ({batch_size} 个, 每行 {cols} 个) ===")
            # 逐行从 buffer 中弹出并打印 batch_size 个元素
            for _ in range(4):
                row_items = []
                for _ in range(cols):
                    if not buffer:
                        break
                    row_items.append(buffer.popleft())
                # 如果某行没有足够元素，也按已有元素打印（保持稳定）
                print("  ".join(map(str, row_items)))
            current_idx += 1

    # 最后若有剩余，按合适列宽打印
    if buffer:
        remaining = len(buffer)
        batch_size = batch_sizes[current_idx]
        if current_idx > 0:
            last_cols = max(1, batch_size // 4)
        else:
            last_cols = min(8, max(1, start_batch // 4))
        print(
            f"\n\n=== 第 {current_idx + 1} 批 ({batch_size} 个, 每行 {last_cols} 个), 剩余 {remaining} 个未满一完整批 ===")
        row = []
        while buffer:
            row.append(buffer.popleft())
            if len(row) >= last_cols:
                print("  ".join(map(str, row)))
                row = []
        if row:
            print("  ".join(map(str, row)))


def dice_feature(dice: list | tuple) -> list:
    """
       分析三个骰子的数值特征。

       参数:
           dice: 包含三个整数（1-6）的列表或元组，代表骰子点数。

       返回:
           list: 一个包含特征字符串的列表。
    for dice in itertools.combinations_with_replacement(range(1,7), r=3)：
        f = dice_feature(dice)
    """
    a, b, c = sorted(dice)  # 排序便于判断连续和相同
    features = []

    # 1. 判断是否为顺子（连续）
    if b - a == 1 and c - b == 1:
        features.append("顺滑")  # 顺子
    # if c - a >= 5:
    #     features.append("夸大")

    # 2. 判断是否三同
    if a == b == c:
        features.append("三相之力")  # 三高

    # # 3. 判断是否两同一异 (Pair)
    # elif a == b or b == c:
    #     # 确定那个成对的数字和单独的数字
    #     features.append("两同")

    # 4. 判断是否为质数组合 (三个点数都是质数)
    # 骰子中的质数面：2, 3, 5
    prime_faces = {2, 3, 5}
    set_dice = set(dice)
    if sum(1 for x in dice if x in prime_faces) >= 2:  # len(set_dice&prime_faces) >= 2
        features.append("精质提升")
    if set_dice == {1, 2, 3}:
        features.append("保底")
    if set_dice == {3, 4}:
        features.append("颠三倒四")
    if set_dice == {4}:
        features.append("四之恶")

    if all(x % 2 == 0 for x in dice):  # 全偶数
        features.append("无独有偶")
    if all(x % 2 == 1 for x in dice):  # 全奇数
        if len(set_dice) > 2:
            features.append("天下无双")  # {1, 3, 5}
    if sum(1 for x in dice if x % 2 == 0) >= 2:
        features.append("好事成双")
        if "顺滑" in features:
            features.append("二段连")

    # 和值大小 (例如，和值大于12算大)
    sum_dice = sum(dice)
    if sum_dice > 16:
        features.append("极限呈现")  # 最大值
    # elif sum_dice < 5:
    #     features.append("极简主义")

    if sum_dice % 3 == 0 and (3 in dice):
        features.append("三生万物")

    if min(dice) > 3:
        features.append("大数倾向")
    if max(dice) < 5:
        if not "三相之力" in features:
            features.append("极简主义")

    # 如果没有显著特征，则标记为“杂色”
    # if not features:
    #     features.append("杂花")

    return features


TRIGGERS = {
    "鼎立": {
        "condition_desc": "三枚骰子点数全部相同，且总和能被 3 整除。",  # ...
        "effect": "登塔层级 ×3（大幅提升挑战层级）。",
        "priority": 100,
        "check": lambda d: (d[0] == d[1] == d[2]) and (sum(d) % 3 == 0)
    },
    "三相之力": {
        "condition_desc": "三枚骰子点数全部相同。",  # 三相共融 1/1/1, 2/2/2, 3/3/3, 4/4/4, 5/5/5, 6/6/6...
        "effect": "同时启用三种塔系的能力与规则（叠加）。",
        "priority": 95,
        "check": lambda d: (d[0] == d[1] == d[2])
    },
    "极简主义": {
        "condition_desc": "三枚骰子点数均小于 5（每枚 ≤4）。",  # ...
        "effect": "早期篇章限制：等级过低无法解锁某些描述或被动（‘未解锁’类效果）。",
        "priority": 30,
        "check": lambda d: max(d) < 5 and not d[0] == d[1] == d[2]
    },
    "大数倾向": {
        "condition_desc": "每个骰子点数都大于 3（每枚 ≥4）。",
        "effect": "所有概率事件的触发概率上升（无论正负事件都更容易发生）。",
        "priority": 40,
        "check": lambda d: min(d) > 3
    },
    "极限呈现": {
        "condition_desc": "三枚骰子总和大于 16。",  # 只有 17 或 18,最大值
        "effect": "所有不稳定属性被抹除下限，按最大可能值呈现（极限表现）。",
        "priority": 110,
        "check": lambda d: sum(d) > 16
    },
    "好事成双": {
        "condition_desc": "至少两枚骰子为偶数面。",  # 例如 2/4/5、4/4/3
        "effect": "选项数 +1 且不消耗选项次数（额外不消耗资源的机会）。",
        "priority": 20,
        "check": lambda d: sum(1 for x in d if x % 2 == 0) >= 2
    },
    "顺势": {
        "condition_desc": "三枚骰子呈现连续点数（顺子）。",  # 例如 1/2/3、3/4/5
        "effect": "战斗技能消耗减半、释放速度提升；短时间内无需升篝火降杀戮值，抗魔与幸运提升。",
        "priority": 60,
        "check": lambda d: (d[1] - d[0] == 1) and (d[2] - d[1] == 1)
    },
    "二段连": {
        "condition_desc": "骰子为顺子，且其中包含两个偶数。",  # 例如 2/3/4
        "effect": "可同时使用任意两种塔系的规则（双塔规则）。",
        "priority": 65,
        "check": lambda d: (d[1] - d[0] == 1) and (d[2] - d[1] == 1) and sum(1 for x in d if x % 2 == 0) >= 2
    },
    "精质提升": {
        "condition_desc": "三枚骰子中至少包含两个质数（质数面为 2、3、5）。",
        "effect": "结算时：一件道具触发品质升阶。",
        "priority": 50,
        "check": lambda d: sum(1 for x in d if x in {2, 3, 5}) >= 2
    },  # 质升
    "天下无双": {
        "condition_desc": "骰子点数为 1、3、5（三个单数且互不相同）。",
        "effect": "大幅提升获取唯一奖励（唯一物品/设施/序列/技能）的概率。",
        "priority": 70,
        "check": lambda d: set(d) == {1, 3, 5}
    },
    "四之恶": {
        "condition_desc": "三枚骰子全部为 4。",  # （4/4/4）
        "effect": "队友死亡后，该使用者继承其所有三塔财产（属性、序列、道具等）。",
        "priority": 120,
        "check": lambda d: set(d) == {4}
    },
    "颠三倒四": {
        "condition_desc": "包含3,4，且其中两个点数相同。",  # 如 4/3/3、3/3/4
        "effect": "可将四次错误行为的损失转为增益，或将三次负面效果转换为正面效果（强力翻盘）。",
        "priority": 55,
        "check": lambda d: set(d) == {3, 4}
    },
    "三界共鸣": {
        "condition_desc": "总和能被 3 整除，且极差大于 3（提高收益但伴随更高风险）。",  # ...
        "effect": "结算时可额外获得三座塔的收益（收益叠加），增加风险但收益显著。",
        "priority": 35,
        "check": lambda d: (sum(d) % 3 == 0) and max(d) - min(d) > 3
    },
    "三生万物": {
        "condition_desc": "三枚骰子中至少有一个为 3，且总和为 3 的倍数。",
        "effect": "当装备数量为 3 或 3 的倍数时，效果进一步增幅（套装或共鸣效果加强）。",
        "priority": 45,
        "check": lambda d: (sum(d) % 3 == 0) and (3 in d)
    },
    "无独有偶": {
        "condition_desc": "点数为双数组合时触发。",  # 例如 4/2/4
        "effect": "当队友人数为偶数时，使用者额外获得与队友同等的奖励；若队友获得唯一序列，则该唯一序列被使用者剥夺（竞争性收益）。",
        "priority": 25,
        "check": lambda d: all(x % 2 == 0 for x in d)
    },
    "保底": {
        "condition_desc": "点数为 1、2、3（顺子 1/2/3）。",
        "effect": "伤害具有保底值且阻止恢复（稳定输出但限制回复效果）。",
        "priority": 90,
        "check": lambda d: set(d) == {1, 2, 3}
    },
}


def dice_feature_game(dice: tuple[int, int, int]) -> list[str]:
    """
    返回该三枚骰子触发的特征列表（以优先级排序，优先级高的先列出）。
    输入 dice 可以是任意顺序的三元组或列表（例如 (3,1,2)）。
    """
    if not (isinstance(dice, (list, tuple)) and len(dice) == 3 and all(1 <= x <= 6 for x in dice)):
        raise ValueError("dice 必须是包含 3 个整数且每个在 1-6 之间。")
    d = tuple(sorted(dice))
    matches = []
    for name, meta in TRIGGERS.items():
        try:
            if meta["check"](d):
                matches.append((meta["priority"], name))
        except Exception:
            # 如果某个检测函数出错，忽略它（保证鲁棒性）
            pass
    # 按 priority 降序排序，然后返回名称
    matches.sort(reverse=True)
    return [name for _, name in matches]


if __name__ == '__main__':
    genotypes_iter = Allele.genotype_iter_by_freq(1000, 360)
    stream_print_genotypes(genotypes_iter)

    rows = []
    for dice in itertools.combinations_with_replacement(range(1, 7), 3):
        features = dice_feature_game(dice)
        rows.append({
            "dice_sorted": dice,
            "dice_str": "/".join(map(str, dice)),
            "features": ", ".join(features) if features else "无"
        })
        print(dice, features, dice_feature(dice))
    print(rows)
