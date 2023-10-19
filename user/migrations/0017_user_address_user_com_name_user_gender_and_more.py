# Generated by Django 4.2.4 on 2023-09-21 11:34

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("user", "0016_remove_user_address_remove_user_com_name_and_more"),
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
            name="high_score",
            field=models.DecimalField(
                decimal_places=2, default=None, max_digits=4, null=True
            ),
        ),
        migrations.AddField(
            model_name="user",
            name="higher_education",
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
            name="phone",
            field=models.IntegerField(default=None, null=True),
        ),
        migrations.AddField(
            model_name="user",
            name="primary_education",
            field=models.TextField(default=None, max_length=200, null=True),
        ),
        migrations.AddField(
            model_name="user",
            name="primary_score",
            field=models.DecimalField(
                decimal_places=2, default=None, max_digits=4, null=True
            ),
        ),
        migrations.AddField(
            model_name="user",
            name="secondary_education",
            field=models.TextField(default=None, max_length=200, null=True),
        ),
        migrations.AddField(
            model_name="user",
            name="secondary_score",
            field=models.DecimalField(
                decimal_places=2, default=None, max_digits=4, null=True
            ),
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
        migrations.AlterField(
            model_name="user",
            name="name",
            field=models.CharField(max_length=255),
        ),
    ]
