#!/usr/bin/env python
# -*- coding: utf-8 -*-
import joblib
import pandas as pd
import os
import numpy as np
import lightgbm as lgb
import tempfile
import shutil

from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from flask import Flask, render_template, request, send_file, after_this_response
from werkzeug.utils import secure_filename

# 初始化 Flask 应用
app = Flask(__name__)

# 使用临时目录处理文件上传
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['ALLOWED_EXTENSIONS'] = {'xlsx', 'xls'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 限制10MB

# 禁用 RDKit 日志
RDLogger.DisableLog('rdApp.error')    
RDLogger.DisableLog('rdApp.warning') 

# 获取当前文件绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 初始化模型和标准化器
model = None
st = None

def load_models():
    """加载预训练模型和标准化器"""
    global model, st
    
    if model is None:
        model_path = os.path.join(BASE_DIR, 'model', 'lgb_model.txt')
        model = lgb.Booster(model_file=model_path)
    
    if st is None:
        st_path = os.path.join(BASE_DIR, 'model', 'std_scaler.bin')
        st = joblib.load(st_path)

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """处理文件上传和预测"""
    # 确保模型已加载
    load_models()
    
    # 检查文件上传
    if 'file' not in request.files:
        return "没有上传文件", 400
    
    file = request.files['file']
    if file.filename == '':
        return "未选择文件", 400
    
    if not allowed_file(file.filename):
        return "无效的文件类型，只支持 Excel (.xlsx, .xls)", 400
    
    # 创建唯一的工作目录
    work_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
    filename = secure_filename(file.filename)
    filepath = os.path.join(work_dir, filename)
    file.save(filepath)
    
    try:
        # 读取Excel数据
        df = pd.read_excel(filepath)
        
        # 检查必要列
        if 'SMILES' not in df.columns:
            return "Excel文件中缺少 'SMILES' 列", 400
        
        # 处理数据并预测
        processed_data = preprocess_data(df)
        X = np.nan_to_num(processed_data)
        X = st.transform(X)
        predictions = model.predict(X)
        
        # 添加预测结果到DataFrame
        df['pregap'] = predictions
        
        # 保存结果
        result_filename = f"result_{filename}"
        result_path = os.path.join(work_dir, result_filename)
        df.to_excel(result_path, index=False)
        
        # 设置响应后清理文件的回调
        @after_this_response
        def cleanup(response):
            """清理临时文件和工作目录"""
            try:
                shutil.rmtree(work_dir)
                app.logger.info(f"清理临时目录: {work_dir}")
            except Exception as e:
                app.logger.error(f"清理文件失败: {str(e)}")
            return response
        
        return send_file(
            result_path,
            as_attachment=True,
            download_name=result_filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
            
    except Exception as e:
        # 发生错误时清理文件
        try:
            shutil.rmtree(work_dir)
        except:
            pass
            
        app.logger.error(f"处理错误: {str(e)}", exc_info=True)
        return f"处理错误: {str(e)}", 500

def get_des(smi):
    """计算分子描述符"""
    if not pd.isna(smi) and smi:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return np.zeros(377)
                
            mol = AllChem.AddHs(mol)
            calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
            ds = np.asarray(calc.CalcDescriptors(mol))
            arr = AllChem.GetMACCSKeysFingerprint(mol)
            arr = np.asarray(arr)
            return np.append(arr, ds)
        except Exception as e:
            app.logger.error(f"处理SMILES错误: {smi}, 错误: {str(e)}")
            return np.zeros(377)
    else:
        return np.zeros(377)

def df2list(df):
    """将DataFrame转换为描述符列表"""
    return pd.DataFrame(df.apply(get_des).values.tolist()) 

def preprocess_data(df):
    """预处理SMILES数据"""
    des_com = df2list(df['SMILES'])
    des_com.reset_index(drop=True, inplace=True)
    return des_com

if __name__ == '__main__':
    # 在生产环境中应禁用debug
    app.run(host='0.0.0.0', port=5000, debug=True)

		