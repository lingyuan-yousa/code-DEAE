# import sys
# from pathlib import Path
#
# # 添加项目根目录到系统路径
# project_root = Path(__file__).parent.parent.parent
# sys.path.append(str(project_root))
# print(f'project root path add：{project_root}')
#
# from fastapi import FastAPI, HTTPException
# from Model.model.model_caller import ModelCaller
# from pydantic import BaseModel
# import logging
#
# app = FastAPI()
#
# # 统一日志配置
# logger = logging.getLogger("API")
# logger.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#
# # 文件日志（绝对路径）
# file_handler = logging.FileHandler(r'e:\\petroleum_logging\\model_training.log')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)
#
# # 控制台日志
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)
#
# class TrainGBDTRequest(BaseModel):
#     n_estimators: int = 100
#     learningRate: float = 0.1
#     maxDepth: int = 3
#     taskID: str = None
#
# class CNNRequest(BaseModel):
#     epochs: int = 100
#     learning_rate: float = 0.001
#
# @app.post("/train/gbdt")
# async def train_gbdt(request: TrainGBDTRequest = None):
#     print("in train_gbdt")
#     if request is None:
#         request = TrainGBDTRequest()
#     try:
#         logger.info(f"Starting training with params: {request.dict() if request else {}}")
#
#         # 初始化调用器
#         caller = ModelCaller(
#             data_path='Model/model/Dataset/2 daqing/daqing1.csv',
#             features=['顶深', '底深', 'SP', 'PE', 'GR', 'AT10', 'AT20', 'AT30', 'AT60', 'AT90', 'AC', 'CNL', 'DEN', 'POR_index', 'Ish'],
#             target_col='LITH'
#         )
#
#         # 执行训练
#         result = caller.call_gbdt(
#             n_estimators=request.n_estimators if request else 100,
#             learning_rate=request.learningRate if request else 0.1,
#             max_depth=request.maxDepth if request else 3,
#             taskID=request.taskID if request else None
#         )
#
#         return {
#             "status": "success",
#             "final_metrics": result['final_metrics'],
#             "metrics_history": result['metrics_history'],
#             "output_files": result['output_files']
#         }
#
#     except FileNotFoundError as e:
#         logger.error(f"File not found: {str(e)}")
#         raise HTTPException(status_code=404, detail=str(e))
#     except ValueError as e:
#         logger.error(f"Invalid parameters: {str(e)}")
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         logger.error(f"Training failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#
# @app.post("/train/cnn")
# async def train_cnn(request : CNNRequest = None):
#     if request is None:
#         request = CNNRequest()
#     try:
#         # 初始化调用器（参数需根据CNN需求调整）
#         caller = ModelCaller(
#             data_path='Model/model/Dataset/2 daqing/daqing1.csv',
#             features=['顶深', '底深', 'SP', 'PE', 'GR', 'AT10', 'AT20', 'AT30', 'AT60', 'AT90', 'AC', 'CNL', 'DEN', 'POR_index', 'Ish'],
#             target_col='LITH'
#         )
#
#         result = caller.call_cnn(
#             epochs=request.epochs if request else 100,
#             learning_rate=request.learning_rate if request else 0.001
#         )
#         return {
#             "status": "success",
#             "final_metrics": result.get('final_metrics', {}),
#             "metrics_history": result.get('metrics_history', []),
#             "output_files": result.get('output_files', [])
#         }
#     except FileNotFoundError as e:
#         logger.error(f"File not found: {str(e)}")
#         raise HTTPException(status_code=404, detail=str(e))
#     except ValueError as e:
#         logger.error(f"Invalid parameters: {str(e)}")
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         logger.error(f"Training failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


import minio
print(minio.__file__)