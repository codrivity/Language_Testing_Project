# Generated by Django 4.2.4 on 2023-09-21 08:24

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("user", "0005_user_fname_user_lname"),
    ]

    operations = [
        migrations.AddField(
            model_name="user",
            name="address",
            field=models.TextField(default=None, max_length=200, null=True),
        ),
        migrations.AddField(
            model_name="user",
            name="com_name",
            field=models.TextField(default=None, max_length=200, null=True),
        ),
        migrations.AddField(
            model_name="user",
            name="gender",
            field=models.TextField(default=None, max_length=6, null=True),
        ),
        migrations.AddField(
            model_name="user",
            name="high_education",
            field=models.TextField(default=None, max_length=200, null=True),
        ),
        migrations.AddField(
            model_name="user",
            name="job_desc",
            field=models.TextField(default=None, max_length=200, null=True),
        ),
        migrations.AddField(
            model_name="user",
            name="job_title",
            field=models.TextField(default=None, max_length=200, null=True),
        ),
        migrations.AddField(
            model_name="user",
            name="pri_education",
            field=models.TextField(default=None, max_length=200, null=True),
        ),
        migrations.AddField(
            model_name="user",
            name="sec_education",
            field=models.TextField(default=None, max_length=200, null=True),
        ),
        migrations.AddField(
            model_name="user",
            name="yoe",
            field=models.TextField(default=None, max_length=200, null=True),
        ),
        migrations.AlterField(
            model_name="user",
            name="email",
            field=models.EmailField(default=None, max_length=254, null=True),
        ),
    ]
