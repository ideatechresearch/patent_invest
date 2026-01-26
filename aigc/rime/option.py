import numpy as np
from enum import Enum, auto
from math import log, sqrt, exp, pi, inf
from scipy.stats import norm
import scipy.optimize as opt


class OptionType(Enum):
    """期权类型枚举"""
    CALL = auto()
    PUT = auto()


class OptionEngine:
    """期权定价基类：包含通用计算工具（IV、Greeks数值求导等）"""

    def __init__(self, s, k, t, r, sigma, q=0.0):
        self.s = s  # 标的资产价格
        self.k = k  # 执行价格
        self.t = t  # 剩余期限 (年化)
        self.r = r  # 无风险利率
        self.sigma = sigma  # 波动率
        self.q = q  # 股息率

    def find_iv(self, market_price, option_type: OptionType, precision=1e-5):
        """二分法寻找隐含波动率 (Implied Volatility)"""
        low, high = 0.0001, 3.0
        for _ in range(100):
            mid = (low + high) / 2
            self.sigma = mid
            price = self.calculate_price(option_type)
            if abs(market_price - price) < precision:
                return mid
            if price < market_price:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    def calculate_price(self, option_type: OptionType):
        """由子类实现具体算法"""
        raise NotImplementedError


# ---------------------------------------------------------
# 欧式期权：Black-Scholes 模型
# ---------------------------------------------------------
class EuropeanBS(OptionEngine):
    """基于 Black-Scholes-Merton 公式计算欧式期权"""

    def _get_d1_d2(self):
        d1 = (log(self.s / self.k) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.t) / (self.sigma * sqrt(self.t))
        d2 = d1 - self.sigma * sqrt(self.t)
        return d1, d2

    def calculate_price(self, option_type: OptionType):
        d1, d2 = self._get_d1_d2()
        if option_type == OptionType.CALL:
            return self.s * exp(-self.q * self.t) * norm.cdf(d1) - self.k * exp(-self.r * self.t) * norm.cdf(d2)

        return self.k * exp(-self.r * self.t) * norm.cdf(-d2) - self.s * exp(-self.q * self.t) * norm.cdf(-d1)

    def greeks(self, option_type: OptionType):
        """解析解计算 Greeks"""
        d1, d2 = self._get_d1_d2()
        pdf_d1 = exp(-0.5 * d1 ** 2) / sqrt(2 * pi)

        delta = exp(-self.q * self.t) * norm.cdf(d1) if option_type == OptionType.CALL else exp(-self.q * self.t) * (
                norm.cdf(d1) - 1)
        gamma = (pdf_d1 * exp(-self.q * self.t)) / (self.s * self.sigma * sqrt(self.t))
        vega = self.s * exp(-self.q * self.t) * sqrt(self.t) * pdf_d1
        return {"Delta": delta, "Gamma": gamma, "Vega": vega}


# ---------------------------------------------------------
# 美式期权：BAW 近似模型 (高效)
# ---------------------------------------------------------
class AmericanBAW(OptionEngine):
    """Barone-Adesi and Whaley 美式期权近似定价"""

    def _find_sx(self, option_type: OptionType):
        """寻找期权提前行权的临界价格 Sx"""
        n = 2.0 * (self.r - self.q) / self.sigma ** 2
        k_val = 2.0 * self.r / (self.sigma ** 2 * (1.0 - exp(-self.r * self.t)))

        def obj_func(sx):
            if sx <= 0: return inf
            # 这里的逻辑是寻找使美式溢价平滑对接临界点的 Sx
            bs = EuropeanBS(sx, self.k, self.t, self.r, self.sigma, self.q)
            price_bs = bs.calculate_price(option_type)
            d1 = (log(sx / self.k) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.t) / (self.sigma * sqrt(self.t))

            if option_type == OptionType.CALL:
                q2 = (1.0 - n + sqrt((n - 1.0) ** 2 + 4.0 * k_val)) / 2.0
                return (price_bs + (1.0 - exp(-self.q * self.t) * norm.cdf(d1)) * sx / q2 - sx + self.k) ** 2
            else:
                q1 = (1.0 - n - sqrt((n - 1.0) ** 2 + 4.0 * k_val)) / 2.0
                return (price_bs - (1.0 - exp(-self.q * self.t) * norm.cdf(-d1)) * sx / q1 + sx - self.k) ** 2

        res = opt.fmin(obj_func, self.s, disp=False)
        return res[0]

    def calculate_price(self, option_type: OptionType):
        bs_price = EuropeanBS(self.s, self.k, self.t, self.r, self.sigma, self.q).calculate_price(option_type)
        sx = self._find_sx(option_type)

        # 计算美式期权相对于欧式期权的提前行权溢价
        n = 2.0 * (self.r - self.q) / self.sigma ** 2
        k_val = 2.0 * self.r / (self.sigma ** 2 * (1.0 - exp(-self.r * self.t)))
        d1 = (log(sx / self.k) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.t) / (self.sigma * sqrt(self.t))

        if option_type == OptionType.CALL:
            if self.s >= sx: return self.s - self.k
            q2 = (1.0 - n + sqrt((n - 1.0) ** 2 + 4.0 * k_val)) / 2.0
            a2 = (sx / q2) * (1.0 - exp(-self.q * self.t) * norm.cdf(d1))
            return bs_price + a2 * (self.s / sx) ** q2
        else:
            if self.s <= sx: return self.k - self.s
            q1 = (1.0 - n - sqrt((n - 1.0) ** 2 + 4.0 * k_val)) / 2.0
            a1 = -(sx / q1) * (1.0 - exp(-self.q * self.t) * norm.cdf(-d1))
            return bs_price + a1 * (self.s / sx) ** q1


# ---------------------------------------------------------
# 通用数值模型：Binomial Tree (二叉树)
# ---------------------------------------------------------
class BinomialTree(OptionEngine):
    """二叉树定价：支持美式/欧式，通过 N 步迭代数值求解"""

    def calculate_price(self, option_type: OptionType, steps=100, is_american: bool = True):
        dt = self.t / steps
        u = exp(self.sigma * sqrt(dt))
        d = 1.0 / u
        a = exp((self.r - self.q) * dt)
        p = (a - d) / (u - d)
        df = exp(-self.r * dt)

        # 初始化末端叶子节点价格
        st_prices = self.s * d ** (np.arange(steps, -1, -1)) * u ** (np.arange(0, steps + 1, 1))

        # 期权末端价值
        if option_type == OptionType.CALL:
            values = np.maximum(0, st_prices - self.k)
        else:
            values = np.maximum(0, self.k - st_prices)

        # 逆向推导
        for j in range(steps - 1, -1, -1):
            values = (p * values[1:] + (1 - p) * values[:-1]) * df
            if is_american:
                # 核心逻辑：在每个节点对比（继续持有价值）与（立即行权价值）
                st_prices = st_prices[:-1] * u  # 回退一级标的价格
                if option_type == OptionType.CALL:
                    values = np.maximum(values, st_prices - self.k)
                else:
                    values = np.maximum(values, self.k - st_prices)
        return values[0]


if __name__ == "__main__":
    # 1. 实例化参数
    params = {
        "s": 100, "k": 100, "t": 0.5, "r": 0.05, "sigma": 0.2, "q": 0.02
    }

    # 2. 欧式定价
    engine_euro = EuropeanBS(**params)
    p_euro = engine_euro.calculate_price(OptionType.CALL)
    print(f"欧式 Greeks: {engine_euro.greeks(OptionType.CALL)}")

    # 3. 美式二叉树定价
    engine_ame = BinomialTree(**params)
    p_ame = engine_ame.calculate_price(OptionType.CALL, steps=200, is_american=True)

    # 4. 反求 IV
    market_price = 7.5
    iv_value = engine_ame.find_iv(market_price, OptionType.CALL)

    print(f"欧式价格: {p_euro:.4f}")
    print(f"美式二叉树价格: {p_ame:.4f}")
    print(f"隐含波动率 美式 IV: {iv_value:.2%}")
