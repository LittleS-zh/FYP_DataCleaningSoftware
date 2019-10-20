from data_cleaning_function.data_cleaning import DataCleaning
from django.shortcuts import render
from django.shortcuts import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.http import FileResponse
import os, sys

dc = DataCleaning()
global float_round
float_round = [20]


def index(request):
    return render(request, "index.html", )


def upload_file(request):
    context = {}
    if request.method == "POST":
        try:
            uploaded_file = request.FILES['document']
            fs = FileSystemStorage()
            name = fs.save(uploaded_file.name, uploaded_file)
            url = fs.url(name)
            # context['url'] = url
            context['name'] = uploaded_file.name
            global file_name
            file_name = uploaded_file.name
            path = "static/DataSet_Read/" + file_name
            dc.read_data_file(path)
            os.remove(path)
        except:
            print("Error happen in upload file.")
    return render(request, "uploadFile.html", context)


def question_search(request):
    return render(request, "questionSearch.html")


def data_cleaning_operation(request):
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame(float_round[0])})


def select_rows(request):
    if request.method == "POST":
        row_ceiling = int(request.POST.get("row_ceiling", None))
        row_floor = int(request.POST.get("row_floor", None))
        dc.select_rows(row_ceiling, row_floor)
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame(float_round[0])})


def select_columns(request):
    if request.method == "POST":
        column_left = int(request.POST.get("column_left", None))
        column_right = int(request.POST.get("column_right", None))
        dc.select_column_position(column_left, column_right)
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame(float_round[0])})


def data_de_duplication(request):
    input_ignore_case = request.POST.get("de_duplication_ignoreCase")
    dc.data_de_duplication(input_ignore_case)
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame(float_round[0])})


def select_by_conditions(request):
    input_column = str(request.POST.get("select_by_condition_input_column", None))
    input_condition_operator = str(request.POST.get("select_by_condition_input_operator", None))
    input_condition_number = float(request.POST.get("select_by_condition_input_number", None))
    dc.select_rows_by_conditions(input_column, input_condition_operator, input_condition_number)
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame(float_round[0])})


def detect_outlier_three_sigma(request):
    if request.method == "POST":
        column_input = str(request.POST.get("detect_outlier_three_sigma", None))
        dc.detect_outlier_three_sigma(column_input)
        input_delete_rows_automatically = request.POST.get("detect_outlier_delete_rows_automatically", None)
        if input_delete_rows_automatically:
            dc.deal_with_outlier(column_input)

    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame(float_round[0])})


def detect_outlier_text(request):
    if request.method == "POST":
        column_input = str(request.POST.get("detect_outlier_text", None))
        dc.detect_outlier_text(column_input)
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame(float_round[0])})


def detect_outlier_all(request):
    if request.method == "POST":
        dc.detect_outlier_all()
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame(float_round[0])})


def text_similarity(request):
    if request.method == "POST":
        column_input = str(request.POST.get("text_similarity", None))
        input_words = str(request.POST.get("input_text", None))
        dc.text_similarity(input_words, column_input)
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame(float_round[0])})


def fill_blank(request):
    if request.method == "POST":
        modification_row = int(request.POST.get("modification_row", None))
        dc.fill_blank(modification_row)
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame(float_round[0])})

def outlier_modification(request):
    if request.method == "POST":
        modification_value = str(request.POST.get("modification_value", None))
        modification_row = int(request.POST.get("modification_row", None))
        dc.outlier_modification(modification_value, modification_row)
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame(float_round[0])})


def single_outlier_delete(request):
    if request.method == "POST":
        modification_row = int(request.POST.get("modification_row", None))
        dc.single_outlier_delete(modification_row)
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame(float_round[0])})


def single_missing_value_modification(request):
    if request.method == "POST":
        modification_row = int(request.POST.get("modification_row", None))
        modification_column = int(request.POST.get("modification_column", None))
        dc.single_outlier_modification(modification_row, modification_column)
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame(float_round[0])})


def single_missing_value_delete(request):
    if request.method == "POST":
        modification_row = int(request.POST.get("modification_row", None))
        dc.single_missing_value_delete(modification_row)
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame(float_round[0])})


def deal_with_outlier(request):
    if request.method == "POST":
        column_input = str(request.POST.get("deal_with_outlier", None))
        dc.deal_with_outlier(column_input)
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame(float_round[0])})


def check_missing(request):
    dc.check_missing()
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame(float_round[0])})


def deal_with_missing_value(request):
    selection = request.GET['deal_with_missing_value']
    dc.deal_with_missing_value(selection)
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame(float_round[0])})


def generate_a_file(request):
    path = "static/DataSet_Write/" + file_name
    dc.write_data_file(path)
    file = open(path, 'rb')
    response = FileResponse(file)
    ContentDisposition = 'attachment;filename=' + '"' + file_name + '"'
    response['Content-Type'] = 'application/form-data'
    response['Content-Disposition'] = ContentDisposition
    return response
    # return render(request, "dataCleaningOperation.html", {"data": dc.get_frame()}, )


def limit_float_point_num(request):
    if request.method == "POST":
        float_point_round = int(request.POST.get("float_point_num", None))
        float_round[0] = float_point_round
        return render(request, "dataCleaningOperation.html", {"data": dc.get_frame(float_round[0])})


def revert(request):
    dc.revert_data_frame()
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame(float_round[0])})


def reset(request):
    dc.reset_data_frame()
    return render(request, "dataCleaningOperation.html", {"data": dc.get_frame(float_round[0])})
