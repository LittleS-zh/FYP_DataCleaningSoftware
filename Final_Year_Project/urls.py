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
    path(r'dataCleaningOperation/', views.data_cleaning_operation, name="dataCleaningOperation"),
    path(r'select_rows/', views.select_rows, name="select_rows"),
    path(r'select_columns/', views.select_columns, name="select_columns"),
    path(r'revert/', views.revert, name="revert"),
    path(r'de_duplication/', views.data_de_duplication, name="de_duplication"),
    path(r'detect_outlier_three_sigma/', views.detect_outlier_three_sigma, name="detect_outlier_three_sigma"),
    path(r'check_missing/', views.check_missing, name="check_missing"),
    path(r'deal_with_missing_value/', views.deal_with_missing_value, name="deal_with_missing_value"),
    path(r'generate_a_file/', views.generate_a_file, name="generate_a_file"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
