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
    if request.method == "POST":
        username = request.POST.get("username", None)
        password = request.POST.get("password", None)
        temp = {"user": username, "pwd": password}
        user_list.append(temp)
        # DataCleaning.readDataFile()
    return render(request, "index.html", {"data": dc.get_frame()})