from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

def update_sheet(question1, question2, question3, question4):
    # 使用你下载的密钥文件
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    SERVICE_ACCOUNT_FILE = '/Users/xiaolong/work/DeepEn_web/user-survery-c8d5eb1575e5.json'

    credentials = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    # 表格 ID 和范围
    SAMPLE_SPREADSHEET_ID = '1n1uqMhtZv_ejJZwvlVf3joEZdezm0xUL2iZuPcaG0DU'
    SAMPLE_RANGE_NAME = 'Sheet1!A:D'

    service = build('sheets', 'v4', credentials=credentials)

    # 要写入的数据
    values = [[question1, question2, question3, question4]]
    body = {
        'values': values
    }
    # 写入数据
    result = service.spreadsheets().values().append(
        spreadsheetId=SAMPLE_SPREADSHEET_ID, range=SAMPLE_RANGE_NAME,
        valueInputOption='USER_ENTERED', body=body).execute()

    print('{0} cells appended.'.format(result \
                                       .get('updates') \
                                       .get('updatedCells')))

# 测试数据
update_sheet('Answer 1', 'Answer 2', 'Answer 3', 'Answer 4')
