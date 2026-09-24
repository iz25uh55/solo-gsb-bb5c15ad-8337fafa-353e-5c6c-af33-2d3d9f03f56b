"""单个可逆反应的理想溶液平衡组成计算。

给定唯一配平后的反应（最小互质正整数系数）、各参与物种的**正**初始
摩尔数、溶液体积 V（L）与浓度平衡常数 Kc，求解使

    Qc(ξ) = ∏生成物 ((n0 + ν·ξ)/V)^ν ÷ ∏反应物 ((n0 − ν·ξ)/V)^ν = Kc

成立的反应进度 ξ（按配平式方向，反应物计量系数前为负、生成物为正）。

关键性质
--------
各物种浓度在保证物料非负的可行区间

    ξ_min = max(−n0_生成物/ν_生成物, −n0_反应物/ν_反应物 < 0 一侧)
    ξ_max = min( n0_反应物/ν_反应物,  n0_生成物/ν_生成物 < 0 一侧)

内，``ln Qc`` 关于 ξ 的导数为

    d ln Qc/dξ = Σ生成物 ν²/(n0+νξ) + Σ反应物 ν²/(n0−νξ) > 0

每一项在开区间内都严格为正，故 Qc 在 (ξ_min, ξ_max) 上**连续严格
递增**，端点极限为 Qc(ξ_min)=0（某生成物耗尽）、Qc(ξ_max)=∞（某
反应物耗尽）——对任意 Kc > 0 恰有一个内点解。其符号由初始反应商
唯一确定，这也是物理上反应自发进行的方向：

* Qc(0) < Kc：唯一根在正向半区间 (0, ξ_max)，ξ* > 0；
* Qc(0) > Kc：唯一根在逆向半区间 (ξ_min, 0)，ξ* < 0（允许负进度）；
* Qc(0) = Kc：ξ* = 0（初始即平衡）。

求解流程
--------
#. 在选定半区间的对数空间做 80 次 IEEE-754 浮点二分，快速（微秒级）
   取得根的近似位置（始终取最终区间中点作为种子，保证结果确定）；
   初始量为正，0 与物料端点均不参与浮点评估，不出现 0/0；
#. 以种子为中心、用有限小数网格在其两侧对称构造一个**以精确有理数
   评估 Qc − Kc** 且跨越精确根的包夹（网格步长按数量级自适应倍增）；
   0 与物料端点可以作为包夹端点，其 Qc 符号（<K 或 >K）由单调性与
   极限虚拟给定、永不评估；
#. 在包夹上做精确有理数二分（只评估严格内部点），直到区间宽度 ≤
   ``EXTENT_TOLERANCE``（1e-10 mol），取中点——该值与精确根的绝对
   误差上界为 1e-10/2 mol，远小于接口承诺的 1e-9 mol；即使真根紧贴
   物料边界（任意接近）也成立；
#. 浮点种子失效（极端数量级输入溢出等）时，从 0 起做纯精确的指数
   探测包夹 + 二分，同样给出严格误差保证（对现实输入规模足够快）。

输入十进制先无损转为 :class:`fractions.Fraction`（复用
:func:`app.stoichiometry.decimal_to_fraction`），精确评估阶段全程
不经过 ``float``，Kc、V、n0 均为精确有理数，因此根的归属判定没有
任何浮点歧义。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Iterable

from .balancer import BalanceError, Species
from .cache import species_key
from .stoichiometry import decimal_to_fraction

# 反应进度结果的绝对误差上界（mol）；二分区间宽度取其一半的一半以下，
# 中点误差再折半，远严于接口承诺的 1e-9。
EXTENT_TOLERANCE = Fraction(1, 10**10)

# 浮点对数空间二分的迭代次数（1e-24 的相对收敛，足以定位精确根）
FLOAT_BISECTION_STEPS = 80

# 网格扩张尝试次数（步长从 1e-12 相对量级起倍增）
GRID_EXPANSION_STEPS = 400

# 极端输入（浮点种子失效）时半区间精确二分的迭代上限
FALLBACK_BISECTION_STEPS = 2000

# 浮点中无法表示的区间端点用此值截断（±1e300 仍远在 double 范围内）
_FLOAT_BOUND_LIMIT = 1e300


@dataclass(frozen=True)
class EquilibriumSolution:
    """平衡求解结果；所有量均为精确 :class:`Fraction`，由业务层渲染。"""

    extent: Fraction
    extent_lower: Fraction
    extent_upper: Fraction
    feasible_lower: Fraction
    feasible_upper: Fraction
    initial_q: Fraction
    initial_moles: dict[str, Fraction]
    equilibrium_moles: dict[str, Fraction]


# ------------------------------------------------------------ 输入解析
def parse_positive(text: str, field: str) -> Decimal:
    """把**严格为正**的有限十进制字符串解析为 :class:`Decimal`。

    与 :func:`app.stoichiometry.parse_moles` 的非负校验不同，平衡接口的
    初始摩尔数、体积与 Kc 均要求严格大于 0（浓度为 0 会使 Qc 在端点外
    无定义，0/0 的体积与 Kc 没有物理意义）。允许科学计数法与下划线。
    """
    value = Decimal(text.strip())
    if not value.is_finite():
        raise ValueError(f"{field} 不能是 NaN 或无穷大")
    if value <= 0:
        raise ValueError(f"{field} 必须为正数")
    return value


# ------------------------------------------------------------ 物料区间
@dataclass(frozen=True)
class _ReactionModel:
    """内部使用的反应模型：规范键 -> (初始摩尔, 配平系数, 有符号计量系数)。"""

    signed_coefficients: dict[str, int]
    initial_moles: dict[str, Fraction]
    volume: Fraction
    kc: Fraction
    reactant_keys: frozenset[str]
    product_keys: frozenset[str]


def _feasible_bounds(model: _ReactionModel) -> tuple[Fraction, Fraction]:
    """按配平系数与初始摩尔计算反应进度的非负物料区间 [ξ_min, ξ_max]。

    对每个物种要求 n0 + s·ξ ≥ 0（s 为有符号计量系数：反应物为负、
    生成物为正）：s < 0 给出 ξ ≤ n0/|s| 的上界，s > 0 给出 ξ ≥ −n0/s
    的下界。各物种初始量严格为正，因此下界严格为负、上界严格为正。
    """
    lower = Fraction(0)
    upper = Fraction(0)
    for key, signed in model.signed_coefficients.items():
        n0 = model.initial_moles[key]
        if signed > 0:
            candidate = -n0 / signed
            if candidate < lower:
                lower = candidate
        else:
            candidate = n0 / (-signed)
            if upper == 0 or candidate < upper:
                upper = candidate
    return lower, upper


# ------------------------------------------------------------ Qc 的精确比较
def _leading_mantissa(n: int, head_digits: int = 15) -> float:
    """取大整数前 ``head_digits`` 位构造 [1, 10) 内的 double 尾数。

    用字符串截取而非 ``float(n) / 10**k``——后者在 n 超过约 309 位时
    会溢出为 inf。误差不超过 10^{-head_digits} 相对量级，仅供对数种子。
    """
    s = str(abs(n))
    if len(s) <= head_digits:
        return float(n) / float(10 ** (len(s) - 1))
    return float(int(s[:head_digits])) / float(10 ** (head_digits - 1))


def _log_fraction(value: Fraction) -> float:
    """Fraction 的自然对数（经 double，用于浮点种子阶段）。

    分子/分母可能超出 double 表示范围，因此分别取对数再相减，避免
    ``float(numerator)`` 溢出；log10 由位数直接得到，尾数在 [1, 10) 内，
    误差为 1e-15 相对量级，对种子定位完全足够。
    """
    if value == 0:
        return -math.inf
    num_digits = len(str(abs(value.numerator)))
    den_digits = len(str(abs(value.denominator)))
    num_mantissa = _leading_mantissa(value.numerator)
    den_mantissa = _leading_mantissa(value.denominator)
    return (
        math.log(num_mantissa)
        - math.log(den_mantissa)
        + math.log(10.0) * (num_digits - den_digits)
    )


def _float_term(
    n0: Fraction, signed: int, xi: float, volume_inverse_log: float
) -> float:
    """单个物种对 ln Qc 的贡献 signed·ln((n0 + signed·ξ)/V)。

    n0 经 :func:`_log_fraction` 转换以规避大整数转 double 的溢出；
    浮点种子只在开区间内部评估，n0 + signed·ξ 恒正，对数有定义。
    """
    # n0 + signed·xi 的对数：取 n0 与 signed·xi 中较大者主导，避免减法溢出
    ln_n0 = _log_fraction(n0)
    ln_signed_xi = math.log(abs(signed)) + math.log(abs(xi)) if xi != 0 else -math.inf
    if xi == 0 or signed * xi <= 0:
        # 消耗项或 ξ=0：(e^{ln_n0} + e^{ln_signed_xi})，第二项为负向或不存在
        if ln_signed_xi == -math.inf:
            ln_concentration = ln_n0 + volume_inverse_log
        else:
            ln_concentration = (
                ln_n0
                + math.log1p(-math.exp(ln_signed_xi - ln_n0))
                + volume_inverse_log
            )
    else:
        # 累加项：logsumexp(ln_n0, ln_signed_xi)
        top = max(ln_n0, ln_signed_xi)
        ln_concentration = (
            top
            + math.log1p(math.exp(-abs(ln_n0 - ln_signed_xi)))
            + volume_inverse_log
        )
    return signed * ln_concentration


def _float_log_q(
    model: _ReactionModel, xi: float, volume_inverse_log: float
) -> float:
    """浮点对数空间内的 ln Qc(ξ) − ln Kc（返回 ln Qc，调用方做差）。"""
    total = 0.0
    for key, signed in model.signed_coefficients.items():
        total += _float_term(
            model.initial_moles[key], signed, xi, volume_inverse_log
        )
    return total


def _clamp_float(value: Fraction) -> float:
    """把精确区间端点安全地转成有限 double（极端值截断到 ±1e300）。"""
    result = float(value)
    if math.isinf(result):
        return math.copysign(_FLOAT_BOUND_LIMIT, result)
    return result


def _float_seed(
    model: _ReactionModel, lower: Fraction, upper: Fraction, log_kc: float
) -> Fraction | None:
    """对数空间浮点二分 80 次，返回近似根（Fraction，未经精确验证）。

    任一步出现非有限值即返回 None，由调用方退化为全区间精确二分。
    """
    lo = _clamp_float(lower)
    hi = _clamp_float(upper)
    volume_inverse_log = -_log_fraction(model.volume)
    try:
        for _ in range(FLOAT_BISECTION_STEPS):
            mid = (lo + hi) * 0.5
            if not math.isfinite(mid):
                return None
            log_q = _float_log_q(model, mid, volume_inverse_log)
            if not math.isfinite(log_q):
                return None
            if log_q < log_kc:
                lo = mid
            else:
                hi = mid
        seed = (lo + hi) * 0.5
    except (OverflowError, ZeroDivisionError, ValueError):
        return None
    if not math.isfinite(seed):
        return None
    return Fraction(Decimal(repr(seed)))


# ------------------------------------------------------------ 精确有理数评估
def _compare_q_kc(model: _ReactionModel, xi: Fraction) -> int:
    """精确比较 Qc(ξ) 与 Kc：返回 -1（Qc<Kc）/ 0（相等）/ 1（Qc>Kc）。

    交叉相乘后直接比较两个由精确有理数幂积组成的整数分式——
    Qc 与 Kc 均为 Fraction，比较本身是一次精确的交叉乘法，
    Fraction 的自动约分把中间值控制在合理规模。仅在可行开区间内部调用。
    """
    qc = Fraction(1)
    for key, signed in model.signed_coefficients.items():
        concentration = (model.initial_moles[key] + signed * xi) / model.volume
        qc *= concentration**signed  # signed<0 时为分母幂
    return (qc > model.kc) - (qc < model.kc)


# ------------------------------------------------------------ 精确根包夹
@dataclass(frozen=True)
class _Bracket:
    """合法包夹 [lo, hi]：Qc(lo) < Kc < Qc(hi)。

    端点可以是物料边界：该处 Qc 为 0 或 ∞，符号由极限**虚拟**保证成立，
    永不参与 :func:`_compare_q_kc` 评估；内部二分点始终严格在开区间内。
    """

    lo: Fraction
    hi: Fraction


def _refine_bracket(
    model: _ReactionModel,
    bracket: _Bracket,
    target: Fraction,
    max_steps: int,
) -> tuple[Fraction, Fraction]:
    """把合法包夹二分到宽度 ≤ target。

    每一步只在严格内部点（非 0、非物料端点）评估 Qc；物料边界端点
    仅以虚拟符号固定。返回包含精确根的最小区间 [a, b]，中点与真根的
    绝对误差 ≤ 区间宽度的一半。
    """
    lo, hi = bracket.lo, bracket.hi
    for _ in range(max_steps):
        if hi - lo <= target:
            return lo, hi
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            return lo, hi
        cmp_mid = _compare_q_kc(model, mid)
        if cmp_mid == 0:
            return mid, mid
        if cmp_mid < 0:
            lo = mid
        else:
            hi = mid
    raise BalanceError(
        "INVALID_REQUEST",
        "平衡组成求解未能在迭代上限内收敛：输入的数量级组合过于极端，"
        "请缩小初始摩尔数、体积或 Kc 的数量级范围后重试",
    )


def _half_interval_fallback(
    model: _ReactionModel,
    low: Fraction,
    high: Fraction,
    target: Fraction,
    max_steps: int,
) -> tuple[Fraction, Fraction]:
    """浮点种子失效时，在半开半区间（一端为 0）内纯精确求解。

    从 0 内侧 target 起向物料边界做指数探测，找到第一个 Q 越过 K 的
    内部点，与 0（内部虚拟端）形成合法包夹；始终找不到（根紧贴物料
    边界）时直接以**物料边界本身**作为虚拟端点包夹，因此即使真根距
    边界任意近，中点误差仍严格 ≤ target/2。
    """
    forward = high != 0
    bound = high if forward else low
    sign = 1 if forward else -1

    last_inside: Fraction | None = None
    step = target
    for _ in range(max_steps):
        point = sign * step
        at_edge = (forward and point >= bound) or (not forward and point <= bound)
        if at_edge:
            # 根在最后一个内部点与物料边界之间；用边界作虚拟端点
            if forward:
                if last_inside is None:
                    last_inside = target
                return _refine_bracket(
                    model,
                    _Bracket(last_inside, bound),
                    target,
                    max_steps,
                )
            if last_inside is None:
                last_inside = -target
            return _refine_bracket(
                model,
                _Bracket(bound, last_inside),
                target,
                max_steps,
            )
        comparison = _compare_q_kc(model, point)
        crossing = (forward and comparison > 0) or (not forward and comparison < 0)
        if crossing:
            if forward:
                origin_side = last_inside if last_inside is not None else target
                return _refine_bracket(
                    model, _Bracket(origin_side, point), target, max_steps
                )
            origin_side = last_inside if last_inside is not None else -target
            return _refine_bracket(
                model, _Bracket(point, origin_side), target, max_steps
            )
        last_inside = point
        step *= 2
    raise BalanceError(
        "INVALID_REQUEST",
        "平衡组成求解未能在迭代上限内收敛：输入的数量级组合过于极端，"
        "请缩小初始摩尔数、体积或 Kc 的数量级范围后重试",
    )


def solve_equilibrium(
    reactants: Iterable[Species],
    products: Iterable[Species],
    coefficients: dict[str, int],
    initial_moles_decimal: dict[str, Decimal],
    volume_decimal: Decimal,
    kc_decimal: Decimal,
) -> EquilibriumSolution:
    """求单个可逆反应的平衡进度与各物种平衡摩尔数。

    参数
    ----
    reactants / products:
        已解析、去重排序（:func:`parse_side`）的两侧物种；
    coefficients:
        以 :func:`species_key` 为键的最小互质正整数配平系数；
    initial_moles_decimal:
        以规范键为键、覆盖**全部**参与物种的**正**初始摩尔 Decimal；
    volume_decimal / kc_decimal:
        严格为正的体积（L）与浓度平衡常数 Decimal。

    返回 :class:`EquilibriumSolution`，进度与摩尔数均为精确 Fraction，
    由业务层按十进制字符串渲染；进度的绝对误差上界见
    :data:`EXTENT_TOLERANCE`（中点 ≤ 其一半）。
    """
    reactants = tuple(reactants)
    products = tuple(products)

    initial_moles = {
        key: decimal_to_fraction(value)
        for key, value in initial_moles_decimal.items()
    }
    volume = decimal_to_fraction(volume_decimal)
    kc = decimal_to_fraction(kc_decimal)

    signed_coefficients: dict[str, int] = {}
    reactant_keys: set[str] = set()
    product_keys: set[str] = set()
    for sp in reactants:
        key = species_key(sp)
        signed_coefficients[key] = -coefficients[key]
        reactant_keys.add(key)
    for sp in products:
        key = species_key(sp)
        signed_coefficients[key] = coefficients[key]
        product_keys.add(key)

    model = _ReactionModel(
        signed_coefficients=signed_coefficients,
        initial_moles=initial_moles,
        volume=volume,
        kc=kc,
        reactant_keys=frozenset(reactant_keys),
        product_keys=frozenset(product_keys),
    )

    lower, upper = _feasible_bounds(model)

    # 初始 Qc（ξ = 0 在开区间内部，初始量全为正，各浓度严格为正）
    initial_q = Fraction(1)
    for key, signed in signed_coefficients.items():
        initial_q *= (initial_moles[key] / volume) ** signed
    if initial_q == kc:
        extent = Fraction(0)
        return EquilibriumSolution(
            extent=extent,
            extent_lower=extent,
            extent_upper=extent,
            feasible_lower=lower,
            feasible_upper=upper,
            initial_q=initial_q,
            initial_moles=dict(initial_moles),
            equilibrium_moles=dict(initial_moles),
        )

    # 由初始反应商确定根的方向（Qc 在全区间严格递增，只有一个根）：
    #   Q0<Kc -> 正向半区间 (0, ξ_max)；Q0>Kc -> 逆向半区间 (ξ_min, 0)
    forward = initial_q < kc
    search_lo, search_hi = (
        (Fraction(0), upper) if forward else (lower, Fraction(0))
    )

    log_kc = _log_fraction(kc)
    seed = _float_seed(model, search_lo, search_hi, log_kc)

    bracket: _Bracket | None = None
    if seed is not None and search_lo < seed < search_hi:
        bracket = _build_grid_bracket(model, seed, search_lo, search_hi, forward)

    if bracket is None:
        # 浮点种子失效（极端数量级）：从 0 起指数探测 + 精确二分
        lo, hi = _half_interval_fallback(
            model, search_lo, search_hi, EXTENT_TOLERANCE, FALLBACK_BISECTION_STEPS
        )
    else:
        lo, hi = _refine_bracket(
            model, bracket, EXTENT_TOLERANCE, FALLBACK_BISECTION_STEPS
        )

    extent = (lo + hi) / 2 if lo != hi else lo

    equilibrium_moles = {
        key: initial_moles[key] + signed * extent
        for key, signed in signed_coefficients.items()
    }
    return EquilibriumSolution(
        extent=extent,
        extent_lower=lo,
        extent_upper=hi,
        feasible_lower=lower,
        feasible_upper=upper,
        initial_q=initial_q,
        initial_moles=dict(initial_moles),
        equilibrium_moles=equilibrium_moles,
    )


def _build_grid_bracket(
    model: _ReactionModel,
    seed: Fraction,
    search_lo: Fraction,
    search_hi: Fraction,
    forward: bool,
) -> _Bracket | None:
    """在浮点种子两侧对称构造跨越精确根的包夹（:class:`_Bracket`）。

    网格以种子为中心、步长从种子相对量级 1e-12 开始（种子通常已精确到
    double ulp，一两步即可包夹），包夹不到则逐步放大步长以容纳极端
    函数斜率；始终使用 :func:`_compare_q_kc` 的精确有理数判定，浮点只
    负责选址。**对称构造**保证结果只依赖种子（浮点区间中点）而不依赖
    浮点二分最终的 lo/hi 归属，同一输入的渲染结果逐位确定。

    半区间的 0 端（Q 符号由 Q0 已知）与物料边界（Qc=0/∞，虚拟符号）
    可以直接作为包夹端点、永不评估，因此根紧贴物料边界时误差仍然严格
    受控。扩张仍未包夹（理论上仅发生在种子完全失效时）返回 None。
    """
    seed_abs = abs(seed)
    scale = max(seed_abs, Fraction(1))
    step = scale / 10**12

    # 0 端的虚拟符号：正向半区间 Q(0)<K(-1)；逆向半区间 Q(0)>K(+1)
    origin_sign = -1 if forward else 1
    material_bound = search_hi if forward else search_lo
    material_sign = 1 if forward else -1

    def endpoint(raw: Fraction) -> tuple[Fraction, int]:
        """把网格点映射为 (包夹端点, Qc 符号)；0/物料边界用虚拟符号。"""
        if forward:
            if raw <= 0:
                return Fraction(0), origin_sign
            if raw >= material_bound:
                return material_bound, material_sign
        else:
            if raw >= 0:
                return Fraction(0), origin_sign
            if raw <= material_bound:
                return material_bound, material_sign
        return raw, _compare_q_kc(model, raw)

    for _ in range(GRID_EXPANSION_STEPS):
        lo, lo_cmp = endpoint(seed - step)
        hi, hi_cmp = endpoint(seed + step)

        if lo == hi and (lo == 0 or lo == material_bound):
            # 种子被极端步长推出半区间：放弃网格，走回退路径
            return None

        def is_interior(point: Fraction) -> bool:
            return (
                (0 < point < material_bound)
                if forward
                else (material_bound < point < 0)
            )

        if lo_cmp == 0 and is_interior(lo):
            return _Bracket(lo, lo)
        if hi_cmp == 0 and is_interior(hi):
            return _Bracket(hi, hi)
        if lo_cmp < 0 < hi_cmp:
            return _Bracket(lo, hi)
        step *= 10
    return None


# ------------------------------------------------------------ 进度渲染
def extent_to_decimal_string(value: Fraction, bound: Fraction) -> str:
    """把平衡进度渲染为十进制字符串，保证绝对误差严格小于 ``bound`` mol。

    平衡进度通常为无限小数，且其数量级可能远大于 1——统一 28 位有效
    数字无法在大数量级下保证 1e-9 mol 的**绝对**误差，因此这里按误差
    上界反推定点小数位数：取 ``bound = 1e-k`` 的 k 位再加 2 个保护位
    （EXTENT_TOLERANCE=1e-10 时为 12 位小数，舍入误差 ≤ 5e-13），
    同时把上下文精度按整数部分放大，使用 ROUND_HALF_EVEN。value 为 0
    时返回 ``"0"``。
    """
    if value == 0:
        return "0"
    # bound 为 10 的负整数次幂时，places = 其负指数 + 2 个保护位
    places = max(0, -int(math.log10(float(bound)))) + 2
    integer_digits = max(
        len(str(abs(value.numerator))) - len(str(value.denominator)) + 1, 1
    )
    precision = max(integer_digits + places + 4, 30)
    with localcontext() as ctx:
        ctx.prec = precision
        decimal_value = Decimal(value.numerator) / Decimal(value.denominator)
        quantized = decimal_value.quantize(Decimal(1).scaleb(-places))
    return _strip_decimal(quantized)


def _strip_decimal(value: Decimal) -> str:
    """Decimal 定点渲染：不带指数、去掉多余尾随零。"""
    sign, digits, exponent = value.as_tuple()
    text = "".join(map(str, digits))
    if exponent >= 0:
        body = text + "0" * exponent
    else:
        places = -exponent
        if places < len(text):
            body = text[:-places] + "." + text[-places:]
        else:
            body = "0." + "0" * (places - len(text)) + text
    if "." in body:
        body = body.rstrip("0").rstrip(".")
    return ("-" if sign and text.strip("0") else "") + (body or "0")
