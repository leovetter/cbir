from django.shortcuts import render_to_response
from django.http import HttpResponse
from rest_framework.renderers import JSONRenderer
from django.template.context_processors import csrf
from django.conf import settings

from CBIR.forms import UploadImageForm
from CBIR.models import Image
from CBIR.serializers import ImageSerializer
from CBIR.services import ImageSearcher

import shutil
import os
import pickle
import numpy as np
import json
# from CBIR.tasks import fit

# from celery.result import AsyncResult

def index(request):

    c = {}
    c.update(csrf(request))

    return render_to_response("index.html", c)

class JSONResponse(HttpResponse):

    def __init__(self, data, **kwargs):

        content = JSONRenderer().render(data)
        kwargs['content_type'] = 'application/json'
        super(JSONResponse, self).__init__(content, **kwargs)

def get_error_rates(request):

    if os.path.exists('training_outputs'):
        nb_epoch = pickle.load(open('training_outputs/nb_epoch.p', 'rb'))
        train_error_rates = pickle.load(open('training_outputs/train_error_rates.p','rb'))
        valid_error_rates = pickle.load(open('training_outputs/valid_error_rates.p', 'rb'))

        return JSONResponse( [nb_epoch, train_error_rates, valid_error_rates])
    else:
        return JSONResponse([[], [], []])

def task_result(request):

    task = AsyncResult(request.GET['taskid'])

    result = task.get() if task.status == 'SUCCESS' else None

    data = {
    'status': task.status,
    'result': result,
    }

    return JSONResponse(data)

def evaluation(request):

    data = json.loads(request.body)
    dataset = str(data['dataset'])
    algos = [ str(algo) for algo in data['algos'] ]
    metrics = [ str(metric) for metric in data['metrics'] ]

    obj = {}
    for metric in metrics:

        curves = []
        for algo in algos:

            name = '_'.join([dataset, algo, metric]) + '.p'
            curve = pickle.load(open(name, 'rb'))
            curves.append(curve)

        obj[metric] = curves

    print(obj)
    return JSONResponse(obj)

def train_network(request):

    task = fit.delay(request.POST['dataset'],  int(request.POST['batch_size']), int(request.POST['max_epoch']),
                   float(request.POST['learning_rate']), float(request.POST['momentum_rate']),
                   float(request.POST['weight_decay']), float(request.POST['lambda_l1']),
                   False
             )

    return JSONResponse({'taskid': task.id})

def get_datasets_name(request):

    datasets_name = os.listdir(settings.DATASET_DIR)

    return JSONResponse(datasets_name[1:])

def images_results(request):

    if request.method == 'POST':

        print(request.POST)
        form = UploadImageForm(request.POST, request.FILES)

        if form.is_valid():

            algo = request.POST['algo']

            img = Image(img=request.FILES['file'])

            if os.path.exists(settings.MEDIA_ROOT+'/'+settings.UPLOAD_PATH):
                shutil.rmtree(settings.MEDIA_ROOT+'/'+settings.UPLOAD_PATH)
            img.save()

            img_searcher = ImageSearcher()
            images = img_searcher.similar_images(img, algo)

            return JSONResponse(images)

        else:

            return JSONResponse({'bad': 'not biv'})
    else:

        return JSONResponse({'options': 'options'})
