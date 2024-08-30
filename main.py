import json
from typing import List, Optional
from coal_mix_opt import coal_mixed_integer_optimization
import numpy as np
from fastapi import FastAPI, applications
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

from log.log import logger
from purchase_opt import purchase_opt_impl
from utils import register_offline_docs

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


class PurchaseOptInput(BaseModel):
    market_coal: list[list[float]]
    stock_coal: list[list[float]]
    ending_inventory: list[float]
    burning_constraint: list[float]
    total_purchase: float
    replace_rate: float


@app.post("/api/coal_mix_opt")
def coal_mix_opt(coal_mix_input: CoalMixInput):
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


@app.post("/api/purchase_opt")
def purchase_opt(purchase_opt_input: PurchaseOptInput):
    try:
        result = purchase_opt_impl(np.array(purchase_opt_input.market_coal),
                                   np.array(purchase_opt_input.stock_coal),
                                   np.array(purchase_opt_input.ending_inventory),
                                   np.array(purchase_opt_input.burning_constraint),
                                   purchase_opt_input.total_purchase,
                                   purchase_opt_input.replace_rate)
        return {"code": 0, "data": {"purchase_opt_result": result.tolist()}, "err_msg": ""}
    except Exception as e:
        logger.error(f"{e}")
        return {"code": -1, "data": {}, "err_msg": f"求解失败, {e}"}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app)
