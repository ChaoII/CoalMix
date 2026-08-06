import json
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, applications
from fastapi.middleware.cors import CORSMiddleware
from openpyxl.compat import deprecated
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

from log.log import logger
from src.auto_sampling import get_automatic_sampling_points
from src.auto_sampling import get_automatic_sampling_regions
from src.auto_sampling import get_automatic_sampling_points_from_regions
from src.auto_sampling import get_automatic_sampling_regions_rolling
from src.coal_mix_opt import coal_mixed_integer_optimization
from src.coal_mix_opt_v2 import coal_mixed_integer_optimization_v2
from src.coal_mix_opt_v3 import coal_mixed_integer_optimization_v3
from src.coal_mix_simple import coal_mixed_optimization_simple
from src.output_opt import output_opt_impl
from src.purchase_opt import purchase_opt_impl
from src.utils import register_offline_docs

register_offline_docs(applications)
# 实例化app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 挂载静态路径将redoc和swagger-ui文件放置在静态路径下
app.mount("/static", StaticFiles(directory="static"), name="static")


class CoalMixSimpleInput(BaseModel):
    # 煤场信息
    coal_info: list[list[float]]
    # 机组约束
    unit_constraint: list[list[float]]
    total_qty: float
    # 最大混煤数
    max_mix_coal: int
    # 寻优目标
    opt_flag: int


class CoalMixInput(BaseModel):
    coal_info: list[list[float]]
    unit_constraint: list[list[float]]
    container_constraint: list[list[float]]
    feeder_capacity: float
    mix_ratio: List[List[int]]
    mutex_coal: List[List[int]]
    standard_coalQty: float
    max_mix_coal: int
    opt_flag: int
    top_k: int


class CoalMixInputV2(BaseModel):
    coal_info: list[list[float]]
    unit_constraint: list[list[list[float]]]
    container_constraint: list
    mix_ratio: List[List[int]]
    coal_quality: List[float]
    mix_coal_num: int
    opt_flag: Optional[int] = 1


class CoalMixInputV3(BaseModel):
    coal_info: list[list[float]]
    unit_constraint: list[list[list[float]]]
    container_constraint: list
    mix_ratio: List[List[int]]
    coal_quality: List[float]
    mix_coal_num: int = 3
    max_scheme_count: int = 3
    opt_flag: Optional[int] = 0


class PurchaseOptInput(BaseModel):
    market_coal: list[list[float]]
    stock_coal: list[list[float]]
    ending_inventory: list[float]
    burning_constraint: list[float]
    total_purchase: float
    replace_rate: float
    max_purchase_kinds: int = 4


class OutputOptInput(BaseModel):
    # 煤仓存煤信息
    container_coal_info: list[list[float]]
    # 出力约束
    output_constraint: list[list[float]]
    # 机组约束
    unit_constraint: list[list[float]]
    # 煤量
    total_qty: float


class AutoSamplingInput(BaseModel):
    # 车长
    car_length: int
    # 车宽
    car_width: int
    # 拉筋区域
    car_lj: list[dict]
    # 可选区域
    car_kxqy: dict


class AutoSamplingInputRegions(AutoSamplingInput):
    # 采样区域
    regions: list[list[int, int]]


class AutoSamplingRollingInput(BaseModel):
    # 当前批次已采编号列表（无需排序，服务端自动去重）
    used: list[int] = []
    # 本车要采点数
    need: int


@app.post("/api/coal_mix_opt_simple")
def _(coal_mix_input: CoalMixSimpleInput):
    s = coal_mix_input.model_dump()
    json.dump(s, open("./coal_mix_input.json", "w"))
    try:
        mix_case, mix_info, mix_price = coal_mixed_optimization_simple(np.array(coal_mix_input.coal_info),
                                                                       np.array(coal_mix_input.unit_constraint),
                                                                       coal_mix_input.total_qty,
                                                                       coal_mix_input.max_mix_coal,
                                                                       coal_mix_input.opt_flag)

        return {"code": 0,
                "data": {"mix_case": mix_case.tolist(), "mix_info": mix_info.tolist(), "mix_price": mix_price},
                "err_msg": ""}
    except Exception as e:
        logger.error(f"{e}")
        return {"code": -1, "data": {}, "err_msg": f"求解失败, {e}"}


@app.post("/api/coal_mix_opt")
def coal_mix_opt(coal_mix_input: CoalMixInput):
    s = coal_mix_input.model_dump()
    json.dump(s, open("./coal_mix_input.json", "w"))
    try:
        mix_case, mix_info, mix_price = coal_mixed_integer_optimization(np.array(coal_mix_input.coal_info),
                                                                        np.array(coal_mix_input.unit_constraint),
                                                                        np.array(coal_mix_input.container_constraint),
                                                                        coal_mix_input.feeder_capacity,
                                                                        np.array(coal_mix_input.mix_ratio, int),
                                                                        coal_mix_input.mutex_coal,
                                                                        coal_mix_input.standard_coalQty,
                                                                        coal_mix_input.max_mix_coal,
                                                                        coal_mix_input.opt_flag,
                                                                        coal_mix_input.top_k)
        return {"code": 0,
                "data": {"mix_case": mix_case.tolist(), "mix_info": mix_info.tolist(), "mix_price": mix_price},
                "err_msg": ""}
    except Exception as e:
        logger.error(f"{e}")
        return {"code": -1, "data": {}, "err_msg": f"求解失败, {e}"}


@app.post("/api/coal_mix_opt_v2")
def coal_mix_opt_v2(coal_mix_input_v2: CoalMixInputV2):
    s = coal_mix_input_v2.model_dump()
    json.dump(s, open("./coal_mix_input_v2.json", "w"))
    try:
        mix_rates, mix_cases, mix_infos, mix_prices = coal_mixed_integer_optimization_v2(
            np.array(coal_mix_input_v2.coal_info),
            np.array(coal_mix_input_v2.unit_constraint),
            np.array(coal_mix_input_v2.container_constraint, dtype=object),
            np.array(coal_mix_input_v2.mix_ratio, int),
            coal_mix_input_v2.coal_quality,
            coal_mix_input_v2.mix_coal_num,
            coal_mix_input_v2.opt_flag)
        return {"code": 0,
                "data": {"mix_rates": mix_rates, "mix_cases": mix_cases, "mix_infos": mix_infos,
                         "mix_prices": mix_prices},
                "err_msg": ""}

    except Exception as e:
        logger.error(f"{e}")
        return {"code": -1, "data": {}, "err_msg": f"求解失败, {e}"}


@app.post("/api/coal_mix_opt_v3")
def coal_mix_opt_v3(coal_mix_input_v3: CoalMixInputV3):
    s = coal_mix_input_v3.model_dump()
    json.dump(s, open("./coal_mix_input_v3.json", "w"))
    try:
        mix_rates, mix_cases, mix_infos, mix_prices = coal_mixed_integer_optimization_v3(
            np.array(coal_mix_input_v3.coal_info),
            np.array(coal_mix_input_v3.unit_constraint),
            np.array(coal_mix_input_v3.container_constraint, dtype=object),
            np.array(coal_mix_input_v3.mix_ratio, int),
            coal_mix_input_v3.coal_quality,
            coal_mix_input_v3.mix_coal_num,
            coal_mix_input_v3.max_scheme_count,
            coal_mix_input_v3.opt_flag)
        return {"code": 0,
                "data": {"mix_rates": mix_rates, "mix_cases": mix_cases, "mix_infos": mix_infos,
                         "mix_prices": mix_prices},
                "err_msg": ""}

    except Exception as e:
        logger.error(f"{e}")
        return {"code": -1, "data": {}, "err_msg": f"求解失败, {e}"}


@app.post("/api/purchase_opt")
def purchase_opt(purchase_opt_input: PurchaseOptInput):
    s = purchase_opt_input.model_dump()
    json.dump(s, open("./purchase_opt_input.json", "w"))
    try:
        purchase_mount, stocking_mount = purchase_opt_impl(np.array(purchase_opt_input.market_coal),
                                                           np.array(purchase_opt_input.stock_coal),
                                                           np.array(purchase_opt_input.ending_inventory),
                                                           np.array(purchase_opt_input.burning_constraint),
                                                           purchase_opt_input.total_purchase,
                                                           purchase_opt_input.replace_rate,
                                                           purchase_opt_input.max_purchase_kinds)
        return {"code": 0,
                "data": {"purchase_mount": purchase_mount.tolist(), "stocking_mount": stocking_mount.tolist()},
                "err_msg": ""}

    except Exception as e:
        logger.error(f"{e}")
        return {"code": -1, "data": {}, "err_msg": f"求解失败, {e}"}


@app.post("/api/output_opt")
def _(output_opt_input: OutputOptInput):
    s = output_opt_input.model_dump()
    json.dump(s, open("./output_opt_input.json", "w"))
    try:
        output = output_opt_impl(np.array(output_opt_input.container_coal_info),
                                 np.array(output_opt_input.output_constraint),
                                 np.array(output_opt_input.unit_constraint),
                                 np.array(output_opt_input.total_qty))
        return {"code": 0, "data": {"output": output.tolist()}, "err_msg": ""}
    except Exception as e:
        logger.error(f"{e}")
        return {"code": -1, "data": {}, "err_msg": f"求解失败, {e}"}


@app.post("/api/auto_sampling", deprecated=True, description="自动生成真实采样点")
def _(auto_sampling_input: AutoSamplingInput):
    s = auto_sampling_input.model_dump()
    json.dump(s, open("./auto_sampling_input.json", "w"))

    length = auto_sampling_input.car_length
    width = auto_sampling_input.car_width
    ljs = []
    for lj in auto_sampling_input.car_lj:
        x0, y0 = lj["p0"]
        x1, y1 = lj["p1"]
        ljs.append([x0, y0, x1, y1])
    kx = [*auto_sampling_input.car_kxqy["p0"], *auto_sampling_input.car_kxqy["p1"]]
    try:
        output = get_automatic_sampling_points(length, width, ljs, kx)
        return {"code": 0, "data": {"points": output[0], "image": output[1]}, "err_msg": ""}
    except Exception as e:
        logger.error(f"{e}")
        return {"code": -1, "data": {}, "err_msg": f"求解失败, {e}"}


@app.get("/api/auto_sampling_regions", deprecated=True, description="获取自动生成采样点的小区信息")
def _():
    try:
        regions = get_automatic_sampling_regions()
        return {"code": 0, "data": {"regions": regions}, "err_msg": ""}
    except Exception as e:
        logger.error(f"{e}")
        return {"code": -1, "data": {}, "err_msg": f"求解失败, {e}"}


@app.post("/api/auto_sampling_points", description="根据采样小区和车辆信息自动生成真实采样点坐标")
def _(auto_sampling_input: AutoSamplingInputRegions):
    s = auto_sampling_input.model_dump()
    json.dump(s, open("./auto_sampling_input.json", "w"))

    length = auto_sampling_input.car_length
    width = auto_sampling_input.car_width
    ljs = []
    for lj in auto_sampling_input.car_lj:
        x0, y0 = lj["p0"]
        x1, y1 = lj["p1"]
        ljs.append([x0, y0, x1, y1])
    kx = [*auto_sampling_input.car_kxqy["p0"], *auto_sampling_input.car_kxqy["p1"]]
    try:
        output = get_automatic_sampling_points_from_regions(length, width, ljs, kx, auto_sampling_input.regions)
        return {"code": 0, "data": {"points": output[0], "image": output[1]}, "err_msg": ""}
    except Exception as e:
        logger.error(f"{e}")
        return {"code": -1, "data": {}, "err_msg": f"求解失败, {e}"}


@app.post("/api/auto_sampling_rolling_regions", description="跨车滚动采样：传 used + need，返回本车应采的编号与采样小区格子")
def _(rolling_input: AutoSamplingRollingInput):
    s = rolling_input.model_dump()
    json.dump(s, open("./auto_sampling_rolling_input.json", "w"))

    used = rolling_input.used or []
    need = rolling_input.need
    try:
        nums, regions = get_automatic_sampling_regions_rolling(used=used, need=need)
        return {"code": 0, "data": {"nums": nums, "regions": regions}, "err_msg": ""}
    except Exception as e:
        logger.error(f"{e}")
        return {"code": -1, "data": {}, "err_msg": f"求解失败, {e}"}
