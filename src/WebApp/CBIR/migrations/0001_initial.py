# -*- coding: utf-8 -*-
# Generated by Django 1.9.1 on 2016-01-14 14:27
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('img', models.ImageField(default=b'', upload_to=b'upload/')),
                ('name', models.CharField(default=b'', max_length=100)),
            ],
        ),
    ]
