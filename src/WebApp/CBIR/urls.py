from django.conf.urls import patterns, url

# from CBIR.views import IndexView
from CBIR import views

urlpatterns = patterns(
    '',

    url(r'^task_result', views.task_result),
    url(r'^get_error_rates', views.get_error_rates),
    url(r'^train_network', views.train_network),
    url(r'^get_datasets_name', views.get_datasets_name),
    url(r'^image', views.images_results),
    url(r'^evaluation', views.evaluation),
    url('^.*$', views.index, name='index'),
)
