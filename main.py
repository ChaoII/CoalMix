import json
from datetime import datetime
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, applications
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

from log.log import logger
from src.coal_mix_opt import coal_mixed_integer_optimization
from src.coal_mix_opt_v2 import coal_mixed_integer_optimization_v2
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


class PurchaseOptInput(BaseModel):
    market_coal: list[list[float]]
    stock_coal: list[list[float]]
    ending_inventory: list[float]
    burning_constraint: list[float]
    total_purchase: float
    replace_rate: float
    max_purchase_kinds: int = 4


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


class TestToolInput(BaseModel):
    start_time: str
    end_time: str


@app.post("/api/test_tool")
def _(test_input: TestToolInput):
    print(test_input.start_time)
    print(test_input.end_time)
    return {"coal_consume": "今日耗煤量8765t"}


@app.get("/api/get_current_time")
def _():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return {"current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
