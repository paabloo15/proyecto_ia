import pandas as pd
import numpy as np
import os
import pynetlogo

netlogo = pynetlogo.NetLogoLink(gui=False, netlogo_home='C:/Program Files/NetLogo 6.2.2')

netlogo.load_model('battle_royale.nlogo')

def ejecutar_partida():
    netlogo.command('setup')
    netlogo.command('repeat 1000 [ go ]')

    salud = netlogo.report('[salud] of turtles')
    recursos = netlogo.report('[recursos] of turtles')
    x = netlogo.report('[xcor] of turtles')
    y = netlogo.report('[ycor] of turtles')
    enemigos_cerca = netlogo.report('[enemigos_cerca] of turtles')
    agresividad = netlogo.report('[agresividad] of turtles')
    perfil = netlogo.report('[perfil] of turtles')
    rendimiento = netlogo.report('[rendimiento] of turtles')

    df = pd.DataFrame({
        'salud': salud,
        'recursos': recursos,
        'x': x,
        'y': y,
        'enemigos_cerca': enemigos_cerca,
        'agresividad': agresividad,
        'perfil': perfil,
        'rendimiento': rendimiento
    })

    return df


datos_totales = pd.DataFrame()

for i in range(1500):  
    print(f"Simulación {i+1}/1500")
    df = ejecutar_partida()
    datos_totales = pd.concat([datos_totales, df], ignore_index=True)

datos_totales.to_csv("datos_battle_royale.csv", index=False)
print("Datos guardados en 'datos_battle_royale.csv'")
