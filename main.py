from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
from py_algorithm import coal_mixed_integer_optimization

# flask格式
app = Flask(__name__)
CORS(app, supports_credentials=True)
# 解决浏览器输出乱码问题
app.config['JSON_AS_ASCII'] = False


# 满足get和post请求
@app.route("/get_best_case", methods=["POST"])
# 代码区域
def get_best_case():
    data = request.get_json(silent=True)
    mix_cases = mix_infos = mix_prices = None
    if data:
        coal_info: list = data["coalInfo"]
        unit_constraint: list = data["unitConstraint"]
        container_constraint: list = data["containerConstraint"]
        feeder_capacity: float = data["feederCapacity"]
        mix_ratio: list = data["mixRatio"]
        mutex_coal: list = data["mutexCoal"]
        standard_coal_qty: float = data["standardCoalQty"]
        max_mix_coal: int = data["maxMixCoal"]
        opt_flag: int = data["optFlag"]
        top_k: int = data["topK"]
        mix_cases, mix_infos, mix_prices = coal_mixed_integer_optimization(np.array(coal_info),
                                                                           np.array(unit_constraint),
                                                                           np.array(container_constraint),
                                                                           feeder_capacity, np.array(mix_ratio),
                                                                           mutex_coal, standard_coal_qty,
                                                                           max_mix_coal, opt_flag, top_k)
    return jsonify({"mixCases": mix_cases, "mix_infos": mix_infos, "mix_prices": mix_prices})


if __name__ == '__main__':
    app.run("0.0.0.0", "5001", debug=True)
