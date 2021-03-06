"""Final_Year_Project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from FYP_codes import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # path('admin/', admin.site.urls),
    path(r'', views.index),
    path(r'index/', views.index, name="index"),
    path(r'uploadFile/', views.upload_file, name="uploadFile"),
    path(r'questionSearch/', views.question_search, name='question_search'),
    path(r'dataCleaningOperation/', views.data_cleaning_operation, name="dataCleaningOperation"),
    path(r'select_rows/', views.select_rows, name="select_rows"),
    path(r'select_columns/', views.select_columns, name="select_columns"),
    path(r'select_by_conditions/', views.select_by_conditions, name="select_by_conditions"),
    path(r'refresh/', views.refresh, name="refresh"),
    path(r'revert/', views.revert, name="revert"),
    path(r'reset/', views.reset, name="reset"),
    path(r'de_duplication/', views.data_de_duplication, name="de_duplication"),
    path(r'detect_outlier_three_sigma/', views.detect_outlier_three_sigma, name="detect_outlier_three_sigma"),
    path(r'detect_outlier_text/', views.detect_outlier_text, name="detect_outlier_text"),
    path(r'detect_outlier_all/', views.detect_outlier_all, name="detect_outlier_all"),
    path(r'text_similarity/', views.text_similarity, name="text_similarity"),
    path(r'text_similarity_for_stack_overflow/', views.text_similarity_for_stack_overflow, name="text_similarity_for_stack_overflow"),
    path(r'deal_with_outlier/', views.deal_with_outlier, name="deal_with_outlier"),
    path(r'check_missing/', views.check_missing, name="check_missing"),
    path(r'deal_with_missing_value/', views.deal_with_missing_value, name="deal_with_missing_value"),
    path(r'generate_a_file/', views.generate_a_file, name="generate_a_file"),
    path(r'limit_float_point_num/', views.limit_float_point_num, name="limit_float_point_num"),
    path(r'fill_blank/', views.fill_blank, name="fill_blank"),
    path(r'outlier_modification/', views.outlier_modification, name="outlier_modification"),
    path(r'single_outlier_delete/', views.single_outlier_delete, name="single_outlier_delete"),
    path(r'single_missing_value_delete/', views.single_missing_value_delete, name="single_missing_value_delete"),
    path(r'single_missing_value_modification/', views.single_missing_value_modification, name="single_missing_value_modification"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
