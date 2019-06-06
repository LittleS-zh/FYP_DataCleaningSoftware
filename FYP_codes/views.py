import BackEnd
from BackEnd import DataCleaning
from django.shortcuts import render
from django.shortcuts import HttpResponse

# Create your views here.
user_list = [
    {"user": "jack", "pwd": "abc"},
    {"user": "tom", "pwd": "ABC"},
]

data_frame_list = DataCleaning.read_data_file("static/DataSet_Read/1 XAGUSD_QPR.csv", ",", "utf8", 0)


def index(request):
    if request.method == "POST":
        username = request.POST.get("username", None)
        password = request.POST.get("password", None)
        temp = {"user": username, "pwd": password}
        user_list.append(temp)
        # DataCleaning.readDataFile()
    return render(request, "index.html", {"data": data_frame_list})