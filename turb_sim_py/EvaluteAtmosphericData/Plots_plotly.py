# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 09:54:21 2022

@author: victoria.lockridge
"""

import pandas as pd
import numpy as np
import os
#import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
from plotly.subplots import make_subplots

dirn = r"C:\Data\JSSAP"
fileA = r"combined_sharpest_images_withAtmos.xlsx"
dirOut = r"C:\Data\JSSAP\AtmosHTMLs"
pathIn = os.path.join(dirn, fileA)
dfcombo = pd.read_excel(pathIn)

colnames = ['date','time','timeSecs', 'range', 'zoom', 'focus', 'imageFn', 'imageHt', 
            'imageWd', 'pixelStep', 'start','stop','obj_size', 'temp', 'humidity', 
            'windSpeed', 'windDir', 'barPressure', 'solarLoad', 'cn2','r0'] 
dfcombo.columns = colnames
dfview = dfcombo.loc[9:,:] # Pulling only Sept 30th data


### PLOT:  Humidity, Temperature, r0 ########################################################
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Scatter(x=dfview.time, y=dfview.temp, name='temperature', mode='lines+markers',
               ))   
fig.add_trace(
     go.Scatter(x=dfview.time, y=dfview.humidity, name='humidity', mode='lines+markers',
     customdata=np.stack((dfview['zoom'], dfview['range'], dfview['temp'], dfview['r0'], dfview['cn2']), axis=-1),
     hovertemplate='<b>Time</b>: %{x}<br>' +
                          '<b>Temperature</b>: %{customdata[2]:,.1f} deg F<br>' +
                          '<b>Humidity</b>: %{y:,.1f}%<br>' +
                          '<b>Zoom</b>: %{customdata[0]}<br>' +
                          '<b>Range</b>: %{customdata[1]:,.0f}m<br>' +
                          '<b>r0</b>: %{customdata[3]:,.4f}<br>' +
                          '<b>Cn2</b>: %{customdata[4]:,.3e}' +                        
                          '<extra></extra>',
        ))
fig.add_trace(
    go.Scatter(x=dfview.time, y=dfview.r0, name='r0', mode='lines+markers'),
    secondary_y=True)
#fig.update_layout(hovermode='x unified')
fig.update_layout(
    hoverlabel=dict(
        bgcolor="white",
        font_size=16,
        font_family="Rockwell"
    ),
    title = 'September 30',
    yaxis2=dict(
        title="r0",
        titlefont=dict(
            color= "#40E0D0"
        ),
        tickfont=dict(
            color="#40E0D0"
        ),
        tickformat=",.4f",
    ))
fig.update_xaxes(title = "Time")
fig.update_yaxes( tickformat=",.1f", secondary_y=False, title = "Temperature/Humdity")
fig.show()
#pathOut = os.path.join(dirOut, r"plot_humidTempR0.html")
#fig.write_html(pathOut)


### PLOT:  Humidity, Temperature, cn2 #########################################################
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Scatter(x=dfview.time, y=dfview.temp, name='temperature', mode='lines+markers',
               )
    )   
fig.add_trace(
     go.Scatter(x=dfview.time, y=dfview.humidity, name='humidity', mode='lines+markers',
     customdata=np.stack((dfview['zoom'], dfview['range'], dfview['temp'], dfview['r0'], dfview['cn2']), axis=-1),
     hovertemplate='<b>Time</b>: %{x}<br>' +
                          '<b>Temperature</b>: %{customdata[2]:,.1f} deg F<br>' +
                          '<b>Humidity</b>: %{y:,.1f}%<br>' +
                          '<b>Zoom</b>: %{customdata[0]}<br>' +
                          '<b>Range</b>: %{customdata[1]:,.0f}m<br>' +
                          '<b>r0</b>: %{customdata[3]:,.4f}<br>' +
                          '<b>Cn2</b>: %{customdata[4]:,.3e}' +                        
                          '<extra></extra>',
        ))
fig.add_trace(
    go.Scatter(x=dfview.time, y=dfview.cn2, name='Cn2', mode='lines+markers'),
    secondary_y=True)
#fig.update_layout(hovermode='x unified')
fig.update_layout(
    hoverlabel=dict(
        bgcolor="white",
        font_size=16,
        font_family="Rockwell"
    ),
    title = 'September 30',
    yaxis2=dict(
        title="Cn2",
        titlefont=dict(
            color= "#40E0D0"
        ),
        tickfont=dict(
            color="#40E0D0"
        ),
        tickformat=",.2e",
        
    ))
fig.update_xaxes(title = "Time")
fig.update_yaxes( tickformat=",.1f", secondary_y=False, title = "Temperature/Humdity")
fig.show()
fig.show()
#pathOut = os.path.join(dirOut, r"plot_humidTempCn2.html")
#fig.write_html(pathOut)

### PLOT:  Cn2 and r0 #######################################################
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
     go.Scatter(x=dfview.time, y=dfview.r0, name='r0', mode='lines+markers',
     customdata=np.stack((dfview['zoom'], dfview['range'], dfview['temp'], dfview['humidity'], dfview['cn2']), axis=-1),
     hovertemplate='<b>Time</b>: %{x}<br>' +
                          '<b>r0</b>: %{y:,.4f}<br>' +
                          '<b>Temperature</b>: %{customdata[2]:,.1f} deg F<br>' +
                          '<b>Humidity</b>: %{customdata[3]:,.1f}%<br>' +
                          '<b>Zoom</b>: %{customdata[0]}<br>' +
                          '<b>Range</b>: %{customdata[1]:,.0f}m<br>' +
                          '<b>Cn2</b>: %{customdata[4]:,.3e}' +                        
                          '<extra></extra>',
        ))
fig.add_trace(
    go.Scatter(x=dfview.time, y=dfview.cn2, name='Cn2', mode='lines+markers',
    
    customdata=np.stack((dfview['zoom'], dfview['range'], dfview['temp'], dfview['humidity'], dfview['r0']), axis=-1),
     hovertemplate='<b>Time</b>: %{x}<br>' +
                          '<b>Cn2</b>: %{y:,.3e}<br>' +
                          '<b>Temperature</b>: %{customdata[2]:,.1f} deg F<br>' +
                          '<b>Humidity</b>: %{customdata[3]:,.1f}%<br>' +
                          '<b>Zoom</b>: %{customdata[0]}<br>' +
                          '<b>Range</b>: %{customdata[1]:,.0f}m<br>' +
                          
                          '<b>r0</b>: %{customdata[4]:,.4f}' +                   
                          '<extra></extra>',
        ),
        secondary_y=True)
#fig.update_layout(hovermode='x unified')
fig.update_layout(
    hoverlabel=dict(
        bgcolor="white",
        font_size=16,
        font_family="Rockwell"
    ),
    title = 'September 30',
    legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
            ),
    yaxis2=dict(
            title="Cn2",
            titlefont=dict(
                color= "#FF0000"
            ),
            tickfont=dict(
                color="#FF0000"
            ),
            tickformat=",.2e",
    ))

fig.update_xaxes(title = "Time")
fig.update_yaxes( tickformat=",.4f",secondary_y=False, title = "r0")
fig.show()
#pathOut = os.path.join(dirOut, r"plot_cn2R0.html")
#fig.write_html(pathOut)

### PLOT:  Cn2 and r0:  Log10 plots #######################################################################
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
     go.Scatter(x=dfview.time, y=dfview.r0, name='r0', mode='lines+markers', 
     customdata=np.stack((dfview['zoom'], dfview['range'], dfview['temp'], dfview['humidity'], dfview['cn2']), axis=-1),
     hovertemplate='<b>Time</b>: %{x}<br>' +
                          '<b>r0</b>: %{y:,.4f}<br>' +
                          '<b>Temperature</b>: %{customdata[2]:,.1f} deg F<br>' +
                          '<b>Humidity</b>: %{customdata[3]:,.1f}%<br>' +
                          '<b>Zoom</b>: %{customdata[0]}<br>' +
                          '<b>Range</b>: %{customdata[1]:,.0f}m<br>' +
                          '<b>Cn2</b>: %{customdata[4]:,.3e}' +                        
                          '<extra></extra>',
        ))
fig.add_trace(
    go.Scatter(x=dfview.time, y=dfview.cn2, name='Cn2', mode='lines+markers',
    
    customdata=np.stack((dfview['zoom'], dfview['range'], dfview['temp'], dfview['humidity'], dfview['r0']), axis=-1),
     hovertemplate='<b>Time</b>: %{x}<br>' +
                          '<b>Cn2</b>: %{y:,.3e}<br>' +
                          '<b>Temperature</b>: %{customdata[2]:,.1f} deg F<br>' +
                          '<b>Humidity</b>: %{customdata[3]:,.1f}%<br>' +
                          '<b>Zoom</b>: %{customdata[0]}<br>' +
                          '<b>Range</b>: %{customdata[1]:,.0f}m<br>' +
                          
                          '<b>r0</b>: %{customdata[4]:,.4f}' +                   
                          '<extra></extra>',
        ),
        secondary_y=True)
#fig.update_layout(hovermode='x unified')
fig.update_layout(
    title = 'September 30',
    hoverlabel=dict(
        bgcolor="white",
        font_size=16,
        font_family="Rockwell"
       ),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
       ),
    xaxis=dict(
    title="Time",
       ),
    yaxis=dict(
        title="log_10(r0)",
        titlefont=dict(
            color="#6495ED"
        ),
        tickfont=dict(
            color="#6495ED"
        ),
        type = "log"
       ),
    yaxis2=dict(
        title="log_10(Cn2)",
        titlefont=dict(
            color= "#FF0000"
        ),
        tickfont=dict(
            color="#FF0000"
        ),
        tickformat = ".2e",

        type = "log"

    ),)

fig.show()
#pathOut = os.path.join(dirOut, r"plot_cn2R0_log.html")
#fig.write_html(pathOut)

