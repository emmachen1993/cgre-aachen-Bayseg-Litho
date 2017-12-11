import numpy as np
import matplotlib


color_names = ["red", "green", "indigo", "orange", "yellow", "brown", "blue", "amber", "pink", "light-blue",
               "lime", "blue-grey", "deep-orange", "grey", "cyan", "deep-purple", "purple", "teal", "light-green"]

color_subnames = ['300', '700', '100', '200', '600', '500', '400', '50', '800', '900']  # 'a100','a200', 'a400', 'a700']


def _create_color_lot(color_names, color_subnames, color_dict_rgb):
    """Returns color [r,g,b] LOT for formation numbers."""
    lot = {}
    i = 0
    for sn in np.arange(len(color_subnames)):
        for n in np.arange(len(color_names)):
            lot[i] = color_dict_rgb[color_names[n]][color_subnames[sn]]
            i += 1

    return lot


color_dict_rgb = {'amber': {'100': [1.0, 0.9254901960784314, 0.7019607843137254],
                            '200': [1.0, 0.8784313725490196, 0.5098039215686274],
                            '300': [1.0, 0.8352941176470589, 0.30980392156862746],
                            '400': [1.0, 0.792156862745098, 0.1568627450980392],
                            '50': [1.0, 0.9725490196078431, 0.8823529411764706],
                            '500': [1.0, 0.7568627450980392, 0.027450980392156862],
                            '600': [1.0, 0.7019607843137254, 0.0],
                            '700': [1.0, 0.6274509803921569, 0.0],
                            '800': [1.0, 0.5607843137254902, 0.0],
                            '900': [1.0, 0.43529411764705883, 0.0],
                            'a100': [1.0, 0.8980392156862745, 0.4980392156862745],
                            'a200': [1.0, 0.8431372549019608, 0.25098039215686274],
                            'a400': [1.0, 0.7686274509803922, 0.0],
                            'a700': [1.0, 0.6705882352941176, 0.0]},
                  'blue': {'100': [0.7333333333333333, 0.8705882352941177, 0.984313725490196],
                           '200': [0.5647058823529412, 0.792156862745098, 0.9764705882352941],
                           '300': [0.39215686274509803, 0.7098039215686275, 0.9647058823529412],
                           '400': [0.25882352941176473, 0.6470588235294118, 0.9607843137254902],
                           '50': [0.8901960784313725, 0.9490196078431372, 0.9921568627450981],
                           '500': [0.12941176470588237, 0.5882352941176471, 0.9529411764705882],
                           '600': [0.11764705882352941, 0.5333333333333333, 0.8980392156862745],
                           '700': [0.09803921568627451, 0.4627450980392157, 0.8235294117647058],
                           '800': [0.08235294117647059, 0.396078431372549, 0.7529411764705882],
                           '900': [0.050980392156862744, 0.2784313725490196, 0.6313725490196078],
                           'a100': [0.5098039215686274, 0.6941176470588235, 1.0],
                           'a200': [0.26666666666666666, 0.5411764705882353, 1.0],
                           'a400': [0.1607843137254902, 0.4745098039215686, 1.0],
                           'a700': [0.1607843137254902, 0.3843137254901961, 1.0]},
                  'blue-grey': {'100': [0.8117647058823529,
                                        0.8470588235294118,
                                        0.8627450980392157],
                                '200': [0.6901960784313725, 0.7450980392156863, 0.7725490196078432],
                                '300': [0.5647058823529412, 0.6431372549019608, 0.6823529411764706],
                                '400': [0.47058823529411764, 0.5647058823529412, 0.611764705882353],
                                '50': [0.9254901960784314, 0.9372549019607843, 0.9450980392156862],
                                '500': [0.3764705882352941, 0.49019607843137253, 0.5450980392156862],
                                '600': [0.32941176470588235, 0.43137254901960786, 0.47843137254901963],
                                '700': [0.27058823529411763, 0.35294117647058826, 0.39215686274509803],
                                '800': [0.21568627450980393, 0.2784313725490196, 0.30980392156862746],
                                '900': [0.14901960784313725, 0.19607843137254902, 0.2196078431372549]},
                  'brown': {'100': [0.8431372549019608, 0.8, 0.7843137254901961],
                            '200': [0.7372549019607844, 0.6666666666666666, 0.6431372549019608],
                            '300': [0.6313725490196078, 0.5333333333333333, 0.4980392156862745],
                            '400': [0.5529411764705883, 0.43137254901960786, 0.38823529411764707],
                            '50': [0.9372549019607843, 0.9215686274509803, 0.9137254901960784],
                            '500': [0.4745098039215686, 0.3333333333333333, 0.2823529411764706],
                            '600': [0.42745098039215684, 0.2980392156862745, 0.2549019607843137],
                            '700': [0.36470588235294116, 0.25098039215686274, 0.21568627450980393],
                            '800': [0.3058823529411765, 0.20392156862745098, 0.1803921568627451],
                            '900': [0.24313725490196078, 0.15294117647058825, 0.13725490196078433]},
                  'cyan': {'100': [0.6980392156862745, 0.9215686274509803, 0.9490196078431372],
                           '200': [0.5019607843137255, 0.8705882352941177, 0.9176470588235294],
                           '300': [0.30196078431372547, 0.8156862745098039, 0.8823529411764706],
                           '400': [0.14901960784313725, 0.7764705882352941, 0.8549019607843137],
                           '50': [0.8784313725490196, 0.9686274509803922, 0.9803921568627451],
                           '500': [0.0, 0.7372549019607844, 0.8313725490196079],
                           '600': [0.0, 0.6745098039215687, 0.7568627450980392],
                           '700': [0.0, 0.592156862745098, 0.6549019607843137],
                           '800': [0.0, 0.5137254901960784, 0.5607843137254902],
                           '900': [0.0, 0.3764705882352941, 0.39215686274509803],
                           'a100': [0.5176470588235295, 1.0, 1.0],
                           'a200': [0.09411764705882353, 1.0, 1.0],
                           'a400': [0.0, 0.8980392156862745, 1.0],
                           'a700': [0.0, 0.7215686274509804, 0.8313725490196079]},
                  'deep-orange': {'100': [1.0, 0.8, 0.7372549019607844],
                                  '200': [1.0, 0.6705882352941176, 0.5686274509803921],
                                  '300': [1.0, 0.5411764705882353, 0.396078431372549],
                                  '400': [1.0, 0.4392156862745098, 0.2627450980392157],
                                  '50': [0.984313725490196, 0.9137254901960784, 0.9058823529411765],
                                  '500': [1.0, 0.3411764705882353, 0.13333333333333333],
                                  '600': [0.9568627450980393, 0.3176470588235294, 0.11764705882352941],
                                  '700': [0.9019607843137255, 0.2901960784313726, 0.09803921568627451],
                                  '800': [0.8470588235294118, 0.2627450980392157, 0.08235294117647059],
                                  '900': [0.7490196078431373, 0.21176470588235294, 0.047058823529411764],
                                  'a100': [1.0, 0.6196078431372549, 0.5019607843137255],
                                  'a200': [1.0, 0.43137254901960786, 0.25098039215686274],
                                  'a400': [1.0, 0.23921568627450981, 0.0],
                                  'a700': [0.8666666666666667, 0.17254901960784313, 0.0]},
                  'deep-purple': {'100': [0.8196078431372549,
                                          0.7686274509803922,
                                          0.9137254901960784],
                                  '200': [0.7019607843137254, 0.615686274509804, 0.8588235294117647],
                                  '300': [0.5843137254901961, 0.4588235294117647, 0.803921568627451],
                                  '400': [0.49411764705882355, 0.3411764705882353, 0.7607843137254902],
                                  '50': [0.9294117647058824, 0.9058823529411765, 0.9647058823529412],
                                  '500': [0.403921568627451, 0.22745098039215686, 0.7176470588235294],
                                  '600': [0.3686274509803922, 0.20784313725490197, 0.6941176470588235],
                                  '700': [0.3176470588235294, 0.17647058823529413, 0.6588235294117647],
                                  '800': [0.27058823529411763, 0.15294117647058825, 0.6274509803921569],
                                  '900': [0.19215686274509805, 0.10588235294117647, 0.5725490196078431],
                                  'a100': [0.7019607843137254, 0.5333333333333333, 1.0],
                                  'a200': [0.48627450980392156, 0.30196078431372547, 1.0],
                                  'a400': [0.396078431372549, 0.12156862745098039, 1.0],
                                  'a700': [0.3843137254901961, 0.0, 0.9176470588235294]},
                  'green': {'100': [0.7843137254901961, 0.9019607843137255, 0.788235294117647],
                            '200': [0.6470588235294118, 0.8392156862745098, 0.6549019607843137],
                            '300': [0.5058823529411764, 0.7803921568627451, 0.5176470588235295],
                            '400': [0.4, 0.7333333333333333, 0.41568627450980394],
                            '50': [0.9098039215686274, 0.9607843137254902, 0.9137254901960784],
                            '500': [0.2980392156862745, 0.6862745098039216, 0.3137254901960784],
                            '600': [0.2627450980392157, 0.6274509803921569, 0.2784313725490196],
                            '700': [0.2196078431372549, 0.5568627450980392, 0.23529411764705882],
                            '800': [0.1803921568627451, 0.49019607843137253, 0.19607843137254902],
                            '900': [0.10588235294117647, 0.3686274509803922, 0.12549019607843137],
                            'a100': [0.7254901960784313, 0.9647058823529412, 0.792156862745098],
                            'a200': [0.4117647058823529, 0.9411764705882353, 0.6823529411764706],
                            'a400': [0.0, 0.9019607843137255, 0.4627450980392157],
                            'a700': [0.0, 0.7843137254901961, 0.3254901960784314]},
                  'grey': {'100': [0.9607843137254902, 0.9607843137254902, 0.9607843137254902],
                           '200': [0.9333333333333333, 0.9333333333333333, 0.9333333333333333],
                           '300': [0.8784313725490196, 0.8784313725490196, 0.8784313725490196],
                           '400': [0.7411764705882353, 0.7411764705882353, 0.7411764705882353],
                           '50': [0.9803921568627451, 0.9803921568627451, 0.9803921568627451],
                           '500': [0.6196078431372549, 0.6196078431372549, 0.6196078431372549],
                           '600': [0.4588235294117647, 0.4588235294117647, 0.4588235294117647],
                           '700': [0.3803921568627451, 0.3803921568627451, 0.3803921568627451],
                           '800': [0.25882352941176473, 0.25882352941176473, 0.25882352941176473],
                           '900': [0.12941176470588237, 0.12941176470588237, 0.12941176470588237]},
                  'indigo': {'100': [0.7725490196078432, 0.792156862745098, 0.9137254901960784],
                             '200': [0.6235294117647059, 0.6588235294117647, 0.8549019607843137],
                             '300': [0.4745098039215686, 0.5254901960784314, 0.796078431372549],
                             '400': [0.3607843137254902, 0.4196078431372549, 0.7529411764705882],
                             '50': [0.9098039215686274, 0.9176470588235294, 0.9647058823529412],
                             '500': [0.24705882352941178, 0.3176470588235294, 0.7098039215686275],
                             '600': [0.2235294117647059, 0.28627450980392155, 0.6705882352941176],
                             '700': [0.18823529411764706, 0.24705882352941178, 0.6235294117647059],
                             '800': [0.1568627450980392, 0.20784313725490197, 0.5764705882352941],
                             '900': [0.10196078431372549, 0.13725490196078433, 0.49411764705882355],
                             'a100': [0.5490196078431373, 0.6196078431372549, 1.0],
                             'a200': [0.3254901960784314, 0.42745098039215684, 0.996078431372549],
                             'a400': [0.23921568627450981, 0.35294117647058826, 0.996078431372549],
                             'a700': [0.18823529411764706, 0.30980392156862746, 0.996078431372549]},
                  'light-blue': {'100': [0.7019607843137254,
                                         0.8980392156862745,
                                         0.9882352941176471],
                                 '200': [0.5058823529411764, 0.8313725490196079, 0.9803921568627451],
                                 '300': [0.30980392156862746, 0.7647058823529411, 0.9686274509803922],
                                 '400': [0.1607843137254902, 0.7137254901960784, 0.9647058823529412],
                                 '50': [0.8823529411764706, 0.9607843137254902, 0.996078431372549],
                                 '500': [0.011764705882352941, 0.6627450980392157, 0.9568627450980393],
                                 '600': [0.011764705882352941, 0.6078431372549019, 0.8980392156862745],
                                 '700': [0.00784313725490196, 0.5333333333333333, 0.8196078431372549],
                                 '800': [0.00784313725490196, 0.4666666666666667, 0.7411764705882353],
                                 '900': [0.00392156862745098, 0.3411764705882353, 0.6078431372549019],
                                 'a100': [0.5019607843137255, 0.8470588235294118, 1.0],
                                 'a200': [0.25098039215686274, 0.7686274509803922, 1.0],
                                 'a400': [0.0, 0.6901960784313725, 1.0],
                                 'a700': [0.0, 0.5686274509803921, 0.9176470588235294]},
                  'light-green': {'100': [0.8627450980392157,
                                          0.9294117647058824,
                                          0.7843137254901961],
                                  '200': [0.7725490196078432, 0.8823529411764706, 0.6470588235294118],
                                  '300': [0.6823529411764706, 0.8352941176470589, 0.5058823529411764],
                                  '400': [0.611764705882353, 0.8, 0.396078431372549],
                                  '50': [0.9450980392156862, 0.9725490196078431, 0.9137254901960784],
                                  '500': [0.5450980392156862, 0.7647058823529411, 0.2901960784313726],
                                  '600': [0.48627450980392156, 0.7019607843137254, 0.25882352941176473],
                                  '700': [0.40784313725490196, 0.6235294117647059, 0.2196078431372549],
                                  '800': [0.3333333333333333, 0.5450980392156862, 0.1843137254901961],
                                  '900': [0.2, 0.4117647058823529, 0.11764705882352941],
                                  'a100': [0.8, 1.0, 0.5647058823529412],
                                  'a200': [0.6980392156862745, 1.0, 0.34901960784313724],
                                  'a400': [0.4627450980392157, 1.0, 0.011764705882352941],
                                  'a700': [0.39215686274509803, 0.8666666666666667, 0.09019607843137255]},
                  'lime': {'100': [0.9411764705882353, 0.9568627450980393, 0.7647058823529411],
                           '200': [0.9019607843137255, 0.9333333333333333, 0.611764705882353],
                           '300': [0.8627450980392157, 0.9058823529411765, 0.4588235294117647],
                           '400': [0.8313725490196079, 0.8823529411764706, 0.3411764705882353],
                           '50': [0.9764705882352941, 0.984313725490196, 0.9058823529411765],
                           '500': [0.803921568627451, 0.8627450980392157, 0.2235294117647059],
                           '600': [0.7529411764705882, 0.792156862745098, 0.2],
                           '700': [0.6862745098039216, 0.7058823529411765, 0.16862745098039217],
                           '800': [0.6196078431372549, 0.615686274509804, 0.1411764705882353],
                           '900': [0.5098039215686274, 0.4666666666666667, 0.09019607843137255],
                           'a100': [0.9568627450980393, 1.0, 0.5058823529411764],
                           'a200': [0.9333333333333333, 1.0, 0.2549019607843137],
                           'a400': [0.7764705882352941, 1.0, 0.0],
                           'a700': [0.6823529411764706, 0.9176470588235294, 0.0]},
                  'orange': {'100': [1.0, 0.8784313725490196, 0.6980392156862745],
                             '200': [1.0, 0.8, 0.5019607843137255],
                             '300': [1.0, 0.7176470588235294, 0.30196078431372547],
                             '400': [1.0, 0.6549019607843137, 0.14901960784313725],
                             '50': [1.0, 0.9529411764705882, 0.8784313725490196],
                             '500': [1.0, 0.596078431372549, 0.0],
                             '600': [0.984313725490196, 0.5490196078431373, 0.0],
                             '700': [0.9607843137254902, 0.48627450980392156, 0.0],
                             '800': [0.9372549019607843, 0.4235294117647059, 0.0],
                             '900': [0.9019607843137255, 0.3176470588235294, 0.0],
                             'a100': [1.0, 0.8196078431372549, 0.5019607843137255],
                             'a200': [1.0, 0.6705882352941176, 0.25098039215686274],
                             'a400': [1.0, 0.5686274509803921, 0.0],
                             'a700': [1.0, 0.42745098039215684, 0.0]},
                  'pink': {'100': [0.9725490196078431, 0.7333333333333333, 0.8156862745098039],
                           '200': [0.9568627450980393, 0.5607843137254902, 0.6941176470588235],
                           '300': [0.9411764705882353, 0.3843137254901961, 0.5725490196078431],
                           '400': [0.9254901960784314, 0.25098039215686274, 0.47843137254901963],
                           '50': [0.9882352941176471, 0.8941176470588236, 0.9254901960784314],
                           '500': [0.9137254901960784, 0.11764705882352941, 0.38823529411764707],
                           '600': [0.8470588235294118, 0.10588235294117647, 0.3764705882352941],
                           '700': [0.7607843137254902, 0.09411764705882353, 0.3568627450980392],
                           '800': [0.6784313725490196, 0.0784313725490196, 0.3411764705882353],
                           '900': [0.5333333333333333, 0.054901960784313725, 0.30980392156862746],
                           'a100': [1.0, 0.5019607843137255, 0.6705882352941176],
                           'a200': [1.0, 0.25098039215686274, 0.5058823529411764],
                           'a400': [0.9607843137254902, 0.0, 0.3411764705882353],
                           'a700': [0.7725490196078432, 0.06666666666666667, 0.3843137254901961]},
                  'purple': {'100': [0.8823529411764706,
                                     0.7450980392156863,
                                     0.9058823529411765],
                             '200': [0.807843137254902, 0.5764705882352941, 0.8470588235294118],
                             '300': [0.7294117647058823, 0.40784313725490196, 0.7843137254901961],
                             '400': [0.6705882352941176, 0.2784313725490196, 0.7372549019607844],
                             '50': [0.9529411764705882, 0.8980392156862745, 0.9607843137254902],
                             '500': [0.611764705882353, 0.15294117647058825, 0.6901960784313725],
                             '600': [0.5568627450980392, 0.1411764705882353, 0.6666666666666666],
                             '700': [0.4823529411764706, 0.12156862745098039, 0.6352941176470588],
                             '800': [0.41568627450980394, 0.10588235294117647, 0.6039215686274509],
                             '900': [0.2901960784313726, 0.0784313725490196, 0.5490196078431373],
                             'a100': [0.9176470588235294, 0.5019607843137255, 0.9882352941176471],
                             'a200': [0.8784313725490196, 0.25098039215686274, 0.984313725490196],
                             'a400': [0.8352941176470589, 0.0, 0.9764705882352941],
                             'a700': [0.6666666666666666, 0.0, 1.0]},
                  'red': {'100': [1.0, 0.803921568627451, 0.8235294117647058],
                          '200': [0.9372549019607843, 0.6039215686274509, 0.6039215686274509],
                          '300': [0.8980392156862745, 0.45098039215686275, 0.45098039215686275],
                          '400': [0.9372549019607843, 0.3254901960784314, 0.3137254901960784],
                          '50': [1.0, 0.9215686274509803, 0.9333333333333333],
                          '500': [0.9568627450980393, 0.2627450980392157, 0.21176470588235294],
                          '600': [0.8980392156862745, 0.2235294117647059, 0.20784313725490197],
                          '700': [0.8274509803921568, 0.1843137254901961, 0.1843137254901961],
                          '800': [0.7764705882352941, 0.1568627450980392, 0.1568627450980392],
                          '900': [0.7176470588235294, 0.10980392156862745, 0.10980392156862745],
                          'a100': [1.0, 0.5411764705882353, 0.5019607843137255],
                          'a200': [1.0, 0.3215686274509804, 0.3215686274509804],
                          'a400': [1.0, 0.09019607843137255, 0.26666666666666666],
                          'a700': [0.8352941176470589, 0.0, 0.0]},
                  'teal': {'100': [0.6980392156862745, 0.8745098039215686, 0.8588235294117647],
                           '200': [0.5019607843137255, 0.796078431372549, 0.7686274509803922],
                           '300': [0.30196078431372547, 0.7137254901960784, 0.6745098039215687],
                           '400': [0.14901960784313725, 0.6509803921568628, 0.6039215686274509],
                           '50': [0.8784313725490196, 0.9490196078431372, 0.9450980392156862],
                           '500': [0.0, 0.5882352941176471, 0.5333333333333333],
                           '600': [0.0, 0.5372549019607843, 0.4823529411764706],
                           '700': [0.0, 0.4745098039215686, 0.4196078431372549],
                           '800': [0.0, 0.4117647058823529, 0.3607843137254902],
                           '900': [0.0, 0.30196078431372547, 0.25098039215686274],
                           'a100': [0.6549019607843137, 1.0, 0.9215686274509803],
                           'a200': [0.39215686274509803, 1.0, 0.8549019607843137],
                           'a400': [0.11372549019607843, 0.9137254901960784, 0.7137254901960784],
                           'a700': [0.0, 0.7490196078431373, 0.6470588235294118]},
                  'yellow': {'100': [1.0, 0.9764705882352941, 0.7686274509803922],
                             '200': [1.0, 0.9607843137254902, 0.615686274509804],
                             '300': [1.0, 0.9450980392156862, 0.4627450980392157],
                             '400': [1.0, 0.9333333333333333, 0.34509803921568627],
                             '50': [1.0, 0.9921568627450981, 0.9058823529411765],
                             '500': [1.0, 0.9215686274509803, 0.23137254901960785],
                             '600': [0.9921568627450981, 0.8470588235294118, 0.20784313725490197],
                             '700': [0.984313725490196, 0.7529411764705882, 0.17647058823529413],
                             '800': [0.9764705882352941, 0.6588235294117647, 0.1450980392156863],
                             '900': [0.9607843137254902, 0.4980392156862745, 0.09019607843137255],
                             'a100': [1.0, 1.0, 0.5529411764705883],
                             'a200': [1.0, 1.0, 0.0],
                             'a400': [1.0, 0.9176470588235294, 0.0],
                             'a700': [1.0, 0.8392156862745098, 0.0]},
                  'black': {'400': [0, 0, 0]},
                  'white': [1, 1, 1]
                  }

color_dict_hex = {
    "red": {
        "50": "#ffebee",
        "100": "#ffcdd2",
        "200": "#ef9a9a",
        "300": "#e57373",
        "400": "#ef5350",
        "500": "#f44336",
        "600": "#e53935",
        "700": "#d32f2f",
        "800": "#c62828",
        "900": "#b71c1c",
        "a100": "#ff8a80",
        "a200": "#ff5252",
        "a400": "#ff1744",
        "a700": "#d50000"
    },
    "pink": {
        "50": "#fce4ec",
        "100": "#f8bbd0",
        "200": "#f48fb1",
        "300": "#f06292",
        "400": "#ec407a",
        "500": "#e91e63",
        "600": "#d81b60",
        "700": "#c2185b",
        "800": "#ad1457",
        "900": "#880e4f",
        "a100": "#ff80ab",
        "a200": "#ff4081",
        "a400": "#f50057",
        "a700": "#c51162"
    },
    "purple": {
        "50": "#f3e5f5",
        "100": "#e1bee7",
        "200": "#ce93d8",
        "300": "#ba68c8",
        "400": "#ab47bc",
        "500": "#9c27b0",
        "600": "#8e24aa",
        "700": "#7b1fa2",
        "800": "#6a1b9a",
        "900": "#4a148c",
        "a100": "#ea80fc",
        "a200": "#e040fb",
        "a400": "#d500f9",
        "a700": "#aa00ff"
    },
    "deep-purple": {
        "50": "#ede7f6",
        "100": "#d1c4e9",
        "200": "#b39ddb",
        "300": "#9575cd",
        "400": "#7e57c2",
        "500": "#673ab7",
        "600": "#5e35b1",
        "700": "#512da8",
        "800": "#4527a0",
        "900": "#311b92",
        "a100": "#b388ff",
        "a200": "#7c4dff",
        "a400": "#651fff",
        "a700": "#6200ea"
    },
    "indigo": {
        "50": "#e8eaf6",
        "100": "#c5cae9",
        "200": "#9fa8da",
        "300": "#7986cb",
        "400": "#5c6bc0",
        "500": "#3f51b5",
        "600": "#3949ab",
        "700": "#303f9f",
        "800": "#283593",
        "900": "#1a237e",
        "a100": "#8c9eff",
        "a200": "#536dfe",
        "a400": "#3d5afe",
        "a700": "#304ffe"
    },
    "blue": {
        "50": "#e3f2fd",
        "100": "#bbdefb",
        "200": "#90caf9",
        "300": "#64b5f6",
        "400": "#42a5f5",
        "500": "#2196f3",
        "600": "#1e88e5",
        "700": "#1976d2",
        "800": "#1565c0",
        "900": "#0d47a1",
        "a100": "#82b1ff",
        "a200": "#448aff",
        "a400": "#2979ff",
        "a700": "#2962ff"
    },
    "light-blue": {
        "50": "#e1f5fe",
        "100": "#b3e5fc",
        "200": "#81d4fa",
        "300": "#4fc3f7",
        "400": "#29b6f6",
        "500": "#03a9f4",
        "600": "#039be5",
        "700": "#0288d1",
        "800": "#0277bd",
        "900": "#01579b",
        "a100": "#80d8ff",
        "a200": "#40c4ff",
        "a400": "#00b0ff",
        "a700": "#0091ea"
    },
    "cyan": {
        "50": "#e0f7fa",
        "100": "#b2ebf2",
        "200": "#80deea",
        "300": "#4dd0e1",
        "400": "#26c6da",
        "500": "#00bcd4",
        "600": "#00acc1",
        "700": "#0097a7",
        "800": "#00838f",
        "900": "#006064",
        "a100": "#84ffff",
        "a200": "#18ffff",
        "a400": "#00e5ff",
        "a700": "#00b8d4"
    },
    "teal": {
        "50": "#e0f2f1",
        "100": "#b2dfdb",
        "200": "#80cbc4",
        "300": "#4db6ac",
        "400": "#26a69a",
        "500": "#009688",
        "600": "#00897b",
        "700": "#00796b",
        "800": "#00695c",
        "900": "#004d40",
        "a100": "#a7ffeb",
        "a200": "#64ffda",
        "a400": "#1de9b6",
        "a700": "#00bfa5"
    },
    "green": {
        "50": "#e8f5e9",
        "100": "#c8e6c9",
        "200": "#a5d6a7",
        "300": "#81c784",
        "400": "#66bb6a",
        "500": "#4caf50",
        "600": "#43a047",
        "700": "#388e3c",
        "800": "#2e7d32",
        "900": "#1b5e20",
        "a100": "#b9f6ca",
        "a200": "#69f0ae",
        "a400": "#00e676",
        "a700": "#00c853"
    },
    "light-green": {
        "50": "#f1f8e9",
        "100": "#dcedc8",
        "200": "#c5e1a5",
        "300": "#aed581",
        "400": "#9ccc65",
        "500": "#8bc34a",
        "600": "#7cb342",
        "700": "#689f38",
        "800": "#558b2f",
        "900": "#33691e",
        "a100": "#ccff90",
        "a200": "#b2ff59",
        "a400": "#76ff03",
        "a700": "#64dd17"
    },
    "lime": {
        "50": "#f9fbe7",
        "100": "#f0f4c3",
        "200": "#e6ee9c",
        "300": "#dce775",
        "400": "#d4e157",
        "500": "#cddc39",
        "600": "#c0ca33",
        "700": "#afb42b",
        "800": "#9e9d24",
        "900": "#827717",
        "a100": "#f4ff81",
        "a200": "#eeff41",
        "a400": "#c6ff00",
        "a700": "#aeea00"
    },
    "yellow": {
        "50": "#fffde7",
        "100": "#fff9c4",
        "200": "#fff59d",
        "300": "#fff176",
        "400": "#ffee58",
        "500": "#ffeb3b",
        "600": "#fdd835",
        "700": "#fbc02d",
        "800": "#f9a825",
        "900": "#f57f17",
        "a100": "#ffff8d",
        "a200": "#ffff00",
        "a400": "#ffea00",
        "a700": "#ffd600"
    },
    "amber": {
        "50": "#fff8e1",
        "100": "#ffecb3",
        "200": "#ffe082",
        "300": "#ffd54f",
        "400": "#ffca28",
        "500": "#ffc107",
        "600": "#ffb300",
        "700": "#ffa000",
        "800": "#ff8f00",
        "900": "#ff6f00",
        "a100": "#ffe57f",
        "a200": "#ffd740",
        "a400": "#ffc400",
        "a700": "#ffab00"
    },
    "orange": {
        "50": "#fff3e0",
        "100": "#ffe0b2",
        "200": "#ffcc80",
        "300": "#ffb74d",
        "400": "#ffa726",
        "500": "#ff9800",
        "600": "#fb8c00",
        "700": "#f57c00",
        "800": "#ef6c00",
        "900": "#e65100",
        "a100": "#ffd180",
        "a200": "#ffab40",
        "a400": "#ff9100",
        "a700": "#ff6d00"
    },
    "deep-orange": {
        "50": "#fbe9e7",
        "100": "#ffccbc",
        "200": "#ffab91",
        "300": "#ff8a65",
        "400": "#ff7043",
        "500": "#ff5722",
        "600": "#f4511e",
        "700": "#e64a19",
        "800": "#d84315",
        "900": "#bf360c",
        "a100": "#ff9e80",
        "a200": "#ff6e40",
        "a400": "#ff3d00",
        "a700": "#dd2c00"
    },
    "brown": {
        "50": "#efebe9",
        "100": "#d7ccc8",
        "200": "#bcaaa4",
        "300": "#a1887f",
        "400": "#8d6e63",
        "500": "#795548",
        "600": "#6d4c41",
        "700": "#5d4037",
        "800": "#4e342e",
        "900": "#3e2723"
    },
    "grey": {
        "50": "#fafafa",
        "100": "#f5f5f5",
        "200": "#eeeeee",
        "300": "#e0e0e0",
        "400": "#bdbdbd",
        "500": "#9e9e9e",
        "600": "#757575",
        "700": "#616161",
        "800": "#424242",
        "900": "#212121"
    },
    "blue-grey": {
        "50": "#eceff1",
        "100": "#cfd8dc",
        "200": "#b0bec5",
        "300": "#90a4ae",
        "400": "#78909c",
        "500": "#607d8b",
        "600": "#546e7a",
        "700": "#455a64",
        "800": "#37474f",
        "900": "#263238"
    }
}

# create dictionary color LOT, e.g. for seaborn use and basis for listed colormap in matplotlib
color_lot = _create_color_lot(color_names, color_subnames, color_dict_rgb)
# listed colormap for matplotlib
bounds = [key for key in color_lot.keys()]
c = []
for key in bounds:
    c.append(color_lot[key])

cmap = matplotlib.colors.ListedColormap(c)
cmap_norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)