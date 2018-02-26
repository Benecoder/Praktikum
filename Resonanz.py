import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np



"""Eigenschwingungen Teilversuch 3"""

Ordnungszahl = np.arange(0,10,1.0)
Frequenzmaxima = [171.6,504,841,1170,1504,1838,2164,2512,2836,3209]
Frequenzmaxima = np.array(Frequenzmaxima,dtype = np.float32)

Frequenzfehler = 3

FrequenzMittel = np.mean(Frequenzmaxima)
OrdnungszahlMittel = np.mean(Ordnungszahl)

plt.errorbar(Ordnungszahl,Frequenzmaxima,yerr = Frequenzfehler,fmt="o")
plt.grid(True)
plt.xlabel("Ordnungszahl")
plt.ylabel("Frequenz in Hz")
plt.show()

"""Resonanz Teilversuch 4"""

Frequenz = [820,800,780,760,740,720,700,680,660,640,620,600,580,560,550,530,
            860,880,900,920,940,960,980,1000,1020,1040,850,830]
Frequenz = np.array(Frequenz,dtype = np.float32)

Frequenzfehler = 3
   
Spannung = [202.3,121.3,100.2,91.9,81.7,69.6,53.7,40.6,
           30.3,24.3,20.7,18.8,18.7,19.6,20.0,30.3,182.4,79.8,49.2,36.8,30.0,25.9,23.4,22.2,
           22.1,23.2,330.0,237.4]

Spannung = np.array(Spannung,dtype = np.float32)

Rauschen = np.array([1.2,1.3,1.2,1.3,1.1])
Rauschen = np.mean(Rauschen)

Spannung -= Rauschen
           
Spannungfehler =  Frequenz*0.05+0.04

plt.errorbar(Frequenz,Spannung,yerr=Spannungfehler,xerr=Frequenzfehler,fmt="o")
plt.grid(True)
plt.ylabel("Spannung in mV")
plt.xlabel("Frequenz in Hz")
plt.show()
