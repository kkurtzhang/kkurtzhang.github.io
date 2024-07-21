---
layout: post
title: Parcel Contractor APP User Manual
date: 2023-09-07 16:23 +1000
category: [Projects,Full Stack Development]
tags: [React Native, Expo]
media_subpath: /parcel-contractor/user-manual/
---

# Parcel Contractor APP User Manual
Welcome, the project aimed at developing a cross-platform mobile app for tracking parcel quantities, built using React Native. We utilized Expo for efficient management and seamless deployment of native iOS and Android applications. This app catered to parcel contractors, enabling them to record and manage daily parcel quantities. It featured an integrated OCR suite for scanning and identifying quantities from scanners, storing the data within the app. Additionally, users could export weekly reports to monitor parcel totals regularly.
This notebook will demonstrate you how to use the Parcel Contractor APP. You can download the APP from [here](https://apps.apple.com/au/app/parcel-contractor/id6451352777).

## Create & Manage account
The APP support the classic email and password login option, social media fast login options(current support Apple and google).

### Classic way to login/signup
{% include embed/video.html src='/SignUp_Classic.MP4' %}
Verification email would be sent to your email:
![Email Verification](/email_verification.jpg)

### Login via Google
{% include embed/video.html src='/signIn_google.MP4' %}

### Login via Apple
{% include embed/video.html src='/signIn_apple.MP4' %}


### Delete account permanently
<span style="color:red">Note: this action can delete all data related to the account, used by your own risk.</span>
{% include embed/video.html src='/delete_account.MP4' %}

## Daily run management
You can add/edit/delete daily number on a specific date, and quickly add express post number at the home page is supported. The APP supports add daily number via camera that leverages the built-in AI OCR tool, which will fill the number automatically.

### Create/Edit daily number, Quickly Add express post number
{% include embed/video.html src='/create_edit_record.MP4' %}

### Create daily number via camera
{% include embed/video.html src='/create_record_ocr.MP4' %}

### Delete daily number
{% include embed/video.html src='/delete_record.MP4' %}

## Perference Setting
The perference setting includes theme change, show weekly earn and pay-day setting. You can set some default values which can be used when you are creating a new daily number.
{% include embed/video.html src='/perference_management.MP4' %}

## Report Management
You can choose any weekly report to be exported as a table, and share it with others.
{% include embed/video.html src='/report_management.MP4' %}

