from flask import Flask, request, redirect
from user_survey import update_sheet

app = Flask(__name__)

@app.route('/submit-survey', methods=['POST'])
def submit_survey():
    # 提取表单数据
    name = request.form['Name']
    email = request.form['Email']
    organization = request.form['Organization']
    reason = request.form['Reason']

    # 调用 user_survey.py 中的函数将数据写入 Google 表格
    update_sheet(name, email, organization, reason)

    # 重定向到下载页面
    return redirect('/static/download-link.html')

if __name__ == '__main__':
    # app.run(debug=True)  # 移除 debug=True 或将其设置为 False
    app.run(host='0.0.0.0', port=5000)  # 添加 host='0.0.0.0' 来允许外部访问
