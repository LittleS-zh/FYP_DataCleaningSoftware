from data_cleaning_function.data_cleaning import DataCleaning
from django.shortcuts import render
from django.shortcuts import HttpResponse

from django.core.files.storage import FileSystemStorage

user_list = [
    {"user": "jack", "pwd": "abc"},
    {"user": "tom", "pwd": "ABC"},
]

dc = DataCleaning()
dc.read_data_file("static/DataSet_Read/1 XAGUSD_QPR.csv")


def index(request):
    return render(request, "index.html", )


def uploadFile(request):
    if request.method == "POST":
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        fs.save(uploaded_file.name, uploaded_file)
        # print(uploaded_file.name)
    return render(request, "uploadFile.html", )


def data_cleaning_operation(request):
    if request.method == "POST":
        row_ceiling = int(request.POST.get("row_ceiling", None))
        row_floor = int(request.POST.get("row_floor", None))
        dc.select_rows(row_ceiling,row_floor)
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame()})
