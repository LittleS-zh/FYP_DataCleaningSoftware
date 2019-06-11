from data_cleaning_function.data_cleaning import DataCleaning
from django.shortcuts import render
from django.shortcuts import HttpResponse
from django.core.files.storage import FileSystemStorage


dc = DataCleaning()
dc.read_data_file("static/DataSet_Read/1 XAGUSD_QPR.csv")


def index(request):
    return render(request, "index.html", )


def upload_file(request):
    context = {}
    if request.method == "POST":
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        url = fs.url(name)
        context['url'] = url
        context['name'] = uploaded_file.name
    return render(request, "uploadFile.html", context)


def data_cleaning_operation(request):
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame()})


def select_rows(request):
    if request.method == "POST":
        row_ceiling = int(request.POST.get("row_ceiling", None))
        row_floor = int(request.POST.get("row_floor", None))
        dc.select_rows(row_ceiling, row_floor)
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame()})


def select_columns(request):
    if request.method == "POST":
        column_left = int(request.POST.get("column_left", None))
        column_right = int(request.POST.get("column_right", None))
        dc.select_column_position(column_left, column_right)
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame()})


def data_de_duplication(request):
    dc.data_de_duplication()
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame()})


def detect_outlier_three_sigma(request):
    if request.method == "POST":
        column_input = str(request.POST.get("column_input", None))
        dc.detect_outlier_three_sigma(column_input)
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame()})


def revert(request):
    dc.revert_data_frame()
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame()})