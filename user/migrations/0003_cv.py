# Generated by Django 4.2.4 on 2023-09-21 06:41

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("user", "0002_alter_user_id"),
    ]

    operations = [
        migrations.CreateModel(
            name="CV",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("cv_file", models.FileField(upload_to="cv_files/")),
            ],
        ),
    ]
