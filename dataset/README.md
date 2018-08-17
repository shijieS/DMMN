## UA-DETRAC dataset

### Organize dataset
1. Download the following items from [official website](http://detrac-db.rit.albany.edu/download) into your *UA-DETRAC root*
- DETRAC-Train-Images 
- DETRAC-Test-Images
- DETRAC-Train-Annotations-XML
- DETRAC-Train-Annotations-MAT:

2. unzip all the downloaded folders, your *UA-DETRAC root* should be like this
```python
"""
 _________________
|--<ua-detrac dataset root>
|  |
|  |- Insight-MVT_Annotation_Test
|  |    |- MVI_39031
|  |    |- ...
|  |    |- MVI_40905
|  |- Insight-MVT_Annotation_Train
|  |    |- MVI_20011
|  |    |- ...
|  |    |- MVI_63563
|  |- DETRAC-Train-Annotations-MAT
|  |    |- MVI_20011.mat
|  |    |- ...
|  |    |- MVI_63563.mat
|  |- DETRAC-Train-Annotations-XML
|  |    |- MVI_20011.xml
|  |    |- ...
|  |    |- MVI_63563.xml
 -----------------
"""

```

3. execute
```commandline
python ./dataset/tools/ua_convert_mat_2_mot.py --ua_root="<UA-DETRAC root>"
```
This will convert the UA mat format tracking result to MOT17 tracking result.

A new folder is created named *DETRAC-Train-Annotations-MOT*