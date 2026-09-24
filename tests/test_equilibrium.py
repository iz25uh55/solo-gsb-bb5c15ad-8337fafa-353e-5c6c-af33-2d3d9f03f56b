"""POST /api/v1/equilibrium/solve 的 HTTP API 与求解核心测试。

测试反应均选用有精确有理数根或可用独立高精度参考核对的方程式
（H2+I2⇌2HI、PCl5⇌PCl3+Cl2、N2+3H2⇌2NH3、Ca(OH)2 中和等），
覆盖正向/逆向/初始平衡三个方向、可行区间边界、极端 Kc、确定性、
误差上界以及全部业务错误路径。
"""

from fractions import Fraction

from decimal import Decimal, localcontext

import pytest
from fastapi.testclient import TestClient

from app.balancer import Species, parse_side, solve_reaction
from app.cache import species_key
from app.equilibrium import (
    EXTENT_TOLERANCE,
    extent_to_decimal_string,
    parse_positive,
    solve_equilibrium,
)
from app.main import app
from app.parser import parse_formula

client = TestClient(app)

PATH = "/api/v1/equilibrium/solve"
ERROR_BOUND = Fraction(1, 2 * 10**10)  # EXTENT_TOLERANCE / 2


def _frac_decimal_error(fraction: Fraction, decimal: Decimal) -> Decimal:
    """Fraction 与 Decimal 的绝对误差（80 位上下文）。"""
    with localcontext() as ctx:
        ctx.prec = 80
        rendered = Decimal(fraction.numerator) / Decimal(fraction.denominator)
        return abs(rendered - decimal)


# ------------------------------------------------------------ 核心：精确算例
def _core_solve(reactants, products, moles, volume, kc):
    """直接调用求解核心（绕过 HTTP），moles 为 {species: 正十进制文本}。"""
    rs = parse_side(reactants, "反应物")
    ps = parse_side(products, "生成物")
    solved = solve_reaction(reactants, products)
    cmap = {
        species_key(sp): coef
        for sp, coef in zip(solved.all_species, solved.coefficients)
    }
    initial = {}
    for name, text in moles.items():
        formula = parse_formula(name)
        sp = Species(raw=name, normalized=formula.normalized, formula=formula)
        initial[species_key(sp)] = Decimal(text)
    return solve_equilibrium(rs, ps, cmap, initial, Decimal(volume), Decimal(kc))


def test_exact_forward_root_hi_reaction():
    # H2 + I2 ⇌ 2HI，各 1 mol、V=1、Kc=64：
    # (1+2x)/(1-x) = 8 -> x = 0.7（精确有限小数）
    sol = _core_solve(
        ["H2", "I2"], ["HI"],
        {"H2": "1", "I2": "1", "HI": "1"}, "1", "64",
    )
    assert sol.extent == Fraction(7, 10)
    assert sol.feasible_lower == Fraction(-1, 2)
    assert sol.feasible_upper == 1
    assert sol.initial_q == 1
    by_key = {
        sp.normalized: sol.equilibrium_moles[species_key(sp)]
        for sp in (*parse_side(["H2", "I2"], "反应物"),
                   *parse_side(["HI"], "生成物"))
    }
    assert by_key["H2"] == Fraction(3, 10)
    assert by_key["I2"] == Fraction(3, 10)
    assert by_key["HI"] == Fraction(12, 5)


def test_exact_reverse_root_negative_extent():
    # Kc = 1/64：(1+2x)/(1-x) = 1/8 -> x = -7/17（负进度，反应逆向）
    sol = _core_solve(
        ["H2", "I2"], ["HI"],
        {"H2": "1", "I2": "1", "HI": "1"}, "1", "0.015625",
    )
    assert sol.extent < 0
    assert abs(sol.extent + Fraction(7, 17)) < ERROR_BOUND
    for moles in sol.equilibrium_moles.values():
        assert moles >= 0


def test_initial_quotient_equals_kc_returns_zero_extent():
    # PCl5 ⇌ PCl3 + Cl2，各 1 mol、V=1 时 Q0 = 1 = Kc -> x=0
    sol = _core_solve(
        ["PCl5"], ["PCl3", "Cl2"],
        {"PCl5": "1", "PCl3": "1", "Cl2": "1"}, "1", "1",
    )
    assert sol.extent == 0
    assert sol.extent_lower == sol.extent_upper == 0
    assert sol.equilibrium_moles == sol.initial_moles

def test_quadratic_root_matches_high_precision_reference():
    # (1+x)^2/(1-x) = 4 -> x = -3 + 2sqrt(3)
    from math import sqrt

    sol = _core_solve(
        ["PCl5"], ["PCl3", "Cl2"],
        {"PCl5": "1", "PCl3": "1", "Cl2": "1"}, "1", "4",
    )
    assert abs(float(sol.extent) - (-3 + 2 * sqrt(3))) < 5e-11


def test_volume_changes_equilibrium_extent():
    # V=2：(1+x)^2/(2(1-x)) = 4 -> x = -5 + 4sqrt(2)
    from math import sqrt

    sol = _core_solve(
        ["PCl5"], ["PCl3", "Cl2"],
        {"PCl5": "1", "PCl3": "1", "Cl2": "1"}, "2", "4",
    )
    assert abs(float(sol.extent) - (-5 + 4 * sqrt(2))) < 5e-11


def test_bracket_width_guarantees_extent_error_bound():
    sol = _core_solve(
        ["Ca(OH)2", "HCl"], ["CaCl2", "H2O"],
        {"Ca(OH)2": "1", "HCl": "2", "CaCl2": "1", "H2O": "5"},
        "2", "1.234567",
    )
    # 最终包夹宽度 ≤ 1e-10，中点绝对误差 ≤ 5e-11 < 接口承诺的 1e-9
    assert sol.extent_upper - sol.extent_lower <= EXTENT_TOLERANCE
    assert sol.extent_lower <= sol.extent <= sol.extent_upper


def test_result_is_deterministic_across_calls():
    # 对称包夹 + 精确二分：同一输入逐位确定（不依赖浮点 lo/hi 归属）
    rendered = {
        extent_to_decimal_string(
            _core_solve(
                ["Ca(OH)2", "HCl"], ["CaCl2", "H2O"],
                {"Ca(OH)2": "1", "HCl": "2", "CaCl2": "1", "H2O": "5"},
                "2", "3.5",
            ).extent,
            EXTENT_TOLERANCE / 2,
        )
        for _ in range(8)
    }
    assert len(rendered) == 1


def test_extreme_kc_still_converges_within_bound():
    for kc, expected in [
        ("1E100", Fraction(1)),       # 近完全正向
        ("1E-100", -Fraction(1, 2)),  # 近完全逆向
        ("1E300", Fraction(1)),
        ("1E-300", -Fraction(1, 2)),
    ]:
        sol = _core_solve(
            ["H2", "I2"], ["HI"],
            {"H2": "1", "I2": "1", "HI": "1"}, "1", kc,
        )
        assert sol.extent_upper - sol.extent_lower <= EXTENT_TOLERANCE
        assert abs(sol.extent - expected) <= ERROR_BOUND
        for moles in sol.equilibrium_moles.values():
            assert moles >= 0


def test_near_limiting_forward_and_reverse_edges():
    # 生成物初始量极小、Kc 极大：进度接近反应物耗尽边界
    sol = _core_solve(
        ["N2", "H2"], ["NH3"],
        {"N2": "1", "H2": "3", "NH3": "1e-9"}, "1", "1000000",
    )
    assert sol.extent > 0
    assert sol.extent < sol.feasible_upper
    for moles in sol.equilibrium_moles.values():
        assert moles >= 0

    # 生成物初始量很大、Kc 极小：进度接近生成物耗尽边界（负）
    sol = _core_solve(
        ["N2", "H2"], ["NH3"],
        {"N2": "1e-9", "H2": "1e-9", "NH3": "3"}, "1", "0.000001",
    )
    assert sol.feasible_lower < sol.extent < 0
    for moles in sol.equilibrium_moles.values():
        assert moles >= 0


def test_parse_positive_rejects_non_positive():
    assert parse_positive("1.5", "x") == Decimal("1.5")
    assert parse_positive("  2e-3 ", "x") == Decimal("2e-3")
    for bad in ["0", "-0", "-1", "NaN", "Infinity", "-Infinity", "abc"]:
        with pytest.raises((ValueError, Exception)):
            parse_positive(bad, "x")


def test_rendered_extent_respects_absolute_error():
    # 大量级进度也必须给出足够的定点小数位（绝对误差，而非相对误差）
    sol = _core_solve(
        ["N2", "H2"], ["NH3"],
        {"N2": "1E20", "H2": "1E20", "NH3": "1E20"}, "1", "1",
    )
    text = extent_to_decimal_string(sol.extent, EXTENT_TOLERANCE / 2)
    assert "." in text and len(text.split(".")[1]) >= 10
    rendered = Decimal(text)
    assert _frac_decimal_error(sol.extent, rendered) <= Decimal(
        "0.00000000005"
    )


# ------------------------------------------------------------ HTTP：成功路径
def _hi_body(kc="64", volume="1"):
    return {
        "reactants": ["H2", "I2"],
        "products": ["HI"],
        "amounts": [
            {"species": "H2", "moles": "1"},
            {"species": "I2", "moles": "1"},
            {"species": "HI", "moles": "1"},
        ],
        "volume_liters": volume,
        "kc": kc,
    }


def test_http_forward_success_payload():
    resp = client.post(PATH, json=_hi_body())
    assert resp.status_code == 200
    data = resp.json()
    assert data["reaction_extent"]["value"] == "0.7"
    assert data["reaction_extent"]["unit"] == "mol"
    assert data["reaction_extent"]["direction"] == "forward"
    assert data["reaction_extent"]["absolute_error_bound"] == "0.0000000001"
    assert data["reaction_extent"]["feasible_range"] == {
        "lower": "-0.5", "upper": "1", "unit": "mol",
    }
    blocks = {x["normalized"]: x for x in data["reactants"] + data["products"]}
    assert blocks["H2"] == {
        "input": "H2", "normalized": "H2", "coefficient": 1,
        "initial_moles": "1", "delta_moles": "-0.7",
        "equilibrium_moles": "0.3",
    }
    assert blocks["HI"]["delta_moles"] == "1.4"
    assert blocks["HI"]["equilibrium_moles"] == "2.4"
    assert data["initial_qc"] == "1"
    assert data["volume_liters"] == "1"
    assert data["kc"] == "64"
    assert data["normalized"]["equation"] == "H2 + I2 -> 2HI"
    assert isinstance(data["reaction_id"], str) and data["reaction_id"]


def test_http_reverse_direction():
    resp = client.post(PATH, json=_hi_body(kc="0.015625"))
    assert resp.status_code == 200
    data = resp.json()
    assert data["reaction_extent"]["direction"] == "reverse"
    value = Decimal(data["reaction_extent"]["value"])
    assert value < 0
    assert abs(value - (-Decimal(7) / Decimal(17))) < Decimal("0.00000000005")


def test_http_at_equilibrium_direction():
    body = {
        "reactants": ["PCl5"],
        "products": ["PCl3", "Cl2"],
        "amounts": [
            {"species": "PCl5", "moles": "1"},
            {"species": "PCl3", "moles": "1"},
            {"species": "Cl2", "moles": "1"},
        ],
        "volume_liters": "1",
        "kc": "1",
    }
    resp = client.post(PATH, json=body)
    assert resp.status_code == 200
    extent = resp.json()["reaction_extent"]
    assert extent["direction"] == "at_equilibrium"
    assert extent["value"] == "0"


def test_http_equivalent_formula_spellings_and_shared_cache():
    # 使用规范化文本一致的等价写法（Ca(OH)2 / Ca [OH]2 / H2 O）
    canonical = {
        "reactants": ["Ca(OH)2", "HCl"],
        "products": ["CaCl2", "H2O"],
        "amounts": [
            {"species": "Ca(OH)2", "moles": "1"},
            {"species": "HCl", "moles": "2"},
            {"species": "CaCl2", "moles": "1"},
            {"species": "H2O", "moles": "5"},
        ],
        "volume_liters": "2",
        "kc": "3.5",
    }
    equivalent = {
        "reactants": ["Ca [OH]2", " HCl"],
        "products": ["CaCl2", "H2 O"],
        "amounts": [
            {"species": "Ca[OH]2", "moles": "1"},
            {"species": " HCl ", "moles": "2"},
            {"species": "CaCl2", "moles": "1"},
            {"species": "H2O", "moles": "5"},
        ],
        "volume_liters": "2",
        "kc": "3.5",
    }
    first = client.post(PATH, json=canonical).json()
    second = client.post(PATH, json=equivalent).json()
    assert first["reaction_id"] == second["reaction_id"]
    assert second["cached"] is True
    assert first["reaction_extent"]["value"] == second["reaction_extent"]["value"]
    # input 字段回显客户端提交物种的原始写法（amounts 中 Ca(OH)2 的等价写法）
    reactants = {x["normalized"]: x for x in second["reactants"]}
    assert reactants["Ca(OH)2"]["input"] == "Ca[OH]2"


def test_http_response_conforms_to_declared_model():
    resp = client.post(PATH, json=_hi_body())
    assert resp.status_code == 200  # response_model 已在路由声明，成功即结构合法
    data = resp.json()
    assert set(data) == {
        "reaction_id", "cached", "normalized", "volume_liters", "kc",
        "initial_qc", "reaction_extent", "reactants", "products",
    }
    extent_keys = {
        "value", "unit", "direction", "absolute_error_bound", "feasible_range",
    }
    assert set(data["reaction_extent"]) == extent_keys


# ------------------------------------------------------------ HTTP：错误路径
def test_cross_side_duplicate_species_conflict():
    body = {
        "reactants": ["H2"],
        "products": ["H2"],
        "amounts": [{"species": "H2", "moles": "1"}],
        "volume_liters": "1",
        "kc": "1",
    }
    resp = client.post(PATH, json=body)
    assert resp.status_code == 409
    error = resp.json()["error"]
    assert error["code"] == "DUPLICATE_SPECIES"
    assert error["details"]["species"] == ["H2"]
    assert "reaction_id" in resp.json()


def test_same_side_duplicate_conflict():
    body = {
        "reactants": ["H2", "H2"],
        "products": ["HI"],
        "amounts": [{"species": "H2", "moles": "1"}],
        "volume_liters": "1",
        "kc": "1",
    }
    assert client.post(PATH, json=body).status_code == 409


def test_reaction_formula_parse_error():
    body = {
        "reactants": ["Xy"],
        "products": ["H2"],
        "amounts": [{"species": "H2", "moles": "1"}],
        "volume_liters": "1",
        "kc": "1",
    }
    resp = client.post(PATH, json=body)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "FORMULA_PARSE_ERROR"


def test_amount_species_parse_error():
    resp = client.post(PATH, json=_hi_body())
    body = _hi_body()
    body["amounts"][0]["species"] = "Xy"
    resp = client.post(PATH, json=body)
    assert resp.status_code == 400
    assert resp.json()["error"]["details"]["field"] == "amounts.species"


def test_non_unique_balance_rejected():
    body = {
        "reactants": ["C", "O2"],
        "products": ["CO", "CO2"],
        "amounts": [
            {"species": "C", "moles": "1"},
            {"species": "O2", "moles": "1"},
            {"species": "CO", "moles": "1"},
            {"species": "CO2", "moles": "1"},
        ],
        "volume_liters": "1",
        "kc": "1",
    }
    resp = client.post(PATH, json=body)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "NON_UNIQUE_BALANCE"


def test_missing_amount_species_aggregated():
    body = _hi_body()
    body["amounts"] = [
        {"species": "H2", "moles": "1"},
        {"species": "I2", "moles": "1"},
    ]
    resp = client.post(PATH, json=body)
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["missing"] == ["HI"]


def test_unknown_and_duplicate_amount_species_aggregated():
    body = _hi_body()
    body["amounts"] = [
        {"species": "H2", "moles": "1"},
        {"species": "H2", "moles": "1"},
        {"species": "I2", "moles": "1"},
        {"species": "HI", "moles": "1"},
        {"species": "O2", "moles": "1"},
    ]
    resp = client.post(PATH, json=body)
    assert resp.status_code == 422
    details = resp.json()["error"]["details"]
    assert details["duplicate"] == ["H2"]
    assert details["unknown"] == ["O2"]


@pytest.mark.parametrize("bad", ["0", "-1", "NaN", "Infinity", "abc"])
def test_non_positive_initial_moles_rejected(bad):
    body = _hi_body()
    body["amounts"][0]["moles"] = bad
    resp = client.post(PATH, json=body)
    assert resp.status_code == 422
    details = resp.json()["error"]["details"]
    assert details["invalid_moles"] == [{"species": "H2", "moles": bad}]


_DETAIL_KEY = {"volume_liters": "invalid_volume", "kc": "invalid_kc"}


@pytest.mark.parametrize("field", ["volume_liters", "kc"])
@pytest.mark.parametrize("bad", ["0", "-2.5", "NaN", "Infinity", ""])
def test_non_positive_volume_and_kc_rejected(field, bad):
    body = _hi_body()
    body[field] = bad
    resp = client.post(PATH, json=body)
    assert resp.status_code == 422
    if bad:
        assert resp.json()["error"]["details"][_DETAIL_KEY[field]] == bad


def test_json_number_instead_of_string_rejected():
    body = _hi_body()
    body["kc"] = 64
    resp = client.post(PATH, json=body)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_missing_required_fields_rejected():
    resp = client.post(PATH, json={"reactants": ["H2"], "products": ["HI"]})
    assert resp.status_code == 422

    body = _hi_body()
    del body["amounts"]
    assert client.post(PATH, json=body).status_code == 422


def test_empty_amounts_rejected():
    body = _hi_body()
    body["amounts"] = []
    assert client.post(PATH, json=body).status_code == 422


def test_multiple_input_problems_aggregated_in_one_response():
    body = {
        "reactants": ["H2", "I2"],
        "products": ["HI"],
        "amounts": [
            {"species": "H2", "moles": "0"},
            {"species": "O2", "moles": "1"},
        ],
        "volume_liters": "-1",
        "kc": "0",
    }
    resp = client.post(PATH, json=body)
    assert resp.status_code == 422
    details = resp.json()["error"]["details"]
    # H2 摩尔数非法，不计入已提供物种
    assert set(details["missing"]) == {"H2", "HI", "I2"}
    assert details["unknown"] == ["O2"]
    assert details["invalid_moles"] == [{"species": "H2", "moles": "0"}]
    assert details["invalid_volume"] == "-1"
    assert details["invalid_kc"] == "0"


# ------------------------------------------------------------ 端到端恒等式核对
def _independent_reference_root(reactants, products, moles_text, volume, kc_text):
    """以 80 位 Decimal 独立二分参考根（按初始 Qc 选半区间）。"""
    rs = parse_side(reactants, "反应物")
    ps = parse_side(products, "生成物")
    solved = solve_reaction(reactants, products)
    cmap = {
        species_key(sp): coef
        for sp, coef in zip(solved.all_species, solved.coefficients)
    }
    n0 = {name: Decimal(text) for name, text in moles_text.items()}

    with localcontext() as ctx:
        ctx.prec = 80
        v = Decimal(volume)
        kc = Decimal(kc_text)

        def quotient(x):
            value = Decimal(1)
            for sp in ps:
                c = Decimal(cmap[species_key(sp)])
                value *= ((n0[sp.normalized] + c * x) / v) ** cmap[species_key(sp)]
            for sp in rs:
                c = Decimal(cmap[species_key(sp)])
                value /= ((n0[sp.normalized] - c * x) / v) ** cmap[species_key(sp)]
            return value

        eps = Decimal("1e-45")
        if quotient(0) < kc:
            upper0 = min(
                Decimal(n0[sp.normalized]) / Decimal(cmap[species_key(sp)])
                for sp in rs
            )
            lo, hi = eps, upper0 - eps
        else:
            lower0 = max(
                -Decimal(n0[sp.normalized]) / Decimal(cmap[species_key(sp)])
                for sp in ps
            )
            lo, hi = lower0 + eps, -eps
        assert quotient(lo) < kc < quotient(hi)
        for _ in range(300):
            mid = (lo + hi) / 2
            if quotient(mid) < kc:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


@pytest.mark.parametrize("reactants,products,moles,volume,kc", [
    (["H2", "I2"], ["HI"],
     {"H2": "1", "I2": "1", "HI": "1"}, "1", "64"),
    (["PCl5"], ["PCl3", "Cl2"],
     {"PCl5": "1", "PCl3": "1", "Cl2": "1"}, "1", "4"),
    (["Ca(OH)2", "HCl"], ["CaCl2", "H2O"],
     {"Ca(OH)2": "1", "HCl": "2", "CaCl2": "1", "H2O": "5"},
     "2", "3.5"),
    (["Cu", "NO3-", "H+"], ["Cu2+", "NO", "H2O"],
     {"Cu": "1", "NO3-": "1", "H+": "1",
      "Cu2+": "1", "NO": "1", "H2O": "1"}, "2", "1.234567"),
])
def test_http_extent_matches_independent_reference(
    reactants, products, moles, volume, kc
):
    body = {
        "reactants": reactants,
        "products": products,
        "amounts": [{"species": name, "moles": text} for name, text in moles.items()],
        "volume_liters": volume,
        "kc": kc,
    }
    resp = client.post(PATH, json=body)
    assert resp.status_code == 200, resp.text
    rendered = Decimal(resp.json()["reaction_extent"]["value"])
    normalized_moles = {parse_formula(n).normalized: t for n, t in moles.items()}
    reference = _independent_reference_root(
        reactants, products, normalized_moles, volume, kc,
    )
    assert abs(rendered - reference) < Decimal("1e-9")
