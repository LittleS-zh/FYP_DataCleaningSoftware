from data_cleaning_function.data_cleaning import DataCleaning
from django.shortcuts import render
from django.shortcuts import HttpResponse

user_list = [
    {"user": "jack", "pwd": "abc"},
    {"user": "tom", "pwd": "ABC"},
]

dc = DataCleaning()
dc.read_data_file("static/DataSet_Read/1 XAGUSD_QPR.csv")


def index(request):
    return render(request, "index.html",)


def uploadFile(request):
    return render(request, "uploadFile.html",)

def dataCleaningOperation(request):
    return render(request, "dataCleaningOperation.html",)