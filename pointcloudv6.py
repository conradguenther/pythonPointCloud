import math
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.colors as mcolors
import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

#-------------------------------------------------------
#----------- Standardwerte / Konstanten ----------------
#-------------------------------------------------------

# Farben für die Geräte definieren: Blau, Rot, Grün und Lila
colors = ['Blues', 'Reds', 'Greens', 'Purples']
colors2 = ['blue', 'red', 'green', 'purple']
# Punktgröße festlegen
point_size = 0.3  # Größe der Punkte

# Pfad zum Ordner mit den CSV-Dateien
input_folder = "input"
output_file = "output/output.csv"

#-------------------------------------------------------
#---------------- Winkel zwischen 2 Vektoren -----------
#-------------------------------------------------------
def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def calcAngle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


#-------------------------------------------------------
#---------------- Ergebnisse abspeichern ---------------
#-------------------------------------------------------

def append_to_csv(new_data):
    """
    Fügt neue Daten zu einer bestehenden CSV-Datei hinzu oder erstellt eine neue Datei, falls sie nicht existiert.

    Parameters:
    - new_data: list[dict] oder pd.DataFrame
        Neue Daten, die angehängt werden sollen.
    - output_file: str
        Pfad zur CSV-Datei, an die die Daten angehängt werden sollen.
    """
    # Sicherstellen, dass new_data ein DataFrame ist
    if isinstance(new_data, list):
        df_new = pd.DataFrame(new_data)
    elif isinstance(new_data, pd.DataFrame):
        df_new = new_data
    else:
        raise ValueError("new_data muss eine Liste von Dictionaries oder ein Pandas DataFrame sein.")

    # Prüfen, ob die Datei existiert
    if os.path.exists(output_file):
        # Datei existiert, neue Daten ohne Header anhängen
        df_new.to_csv(output_file, mode='a', index=False, header=False)
    else:
        # Datei existiert nicht, neue Datei mit Header erstellen
        df_new.to_csv(output_file, mode='w', index=False, header=True)

    print(f"Neue Daten wurden an {output_file} angehängt.")


#-------------------------------------------------------
#----------- Schleife durch die Inputs -----------------
#-------------------------------------------------------

# Alle Dateien im Ordner durchgehen
for filename in os.listdir(input_folder):
  # Nur CSV-Dateien auswählen
  if filename.endswith(".csv"):
    print("filename: " + filename)

    # CSV-Datei laden
    file_path = os.path.join(input_folder, filename)
    df = pd.read_csv(file_path)

    # 3D-Plot erstellen
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 

    devices = df['device'].unique()

    radi = []
    normal_vectors = []
    circle_centers = []

    # Für jedes Gerät eine separate Punktwolke hinzufügen
    for i, device in enumerate(devices):
      # Daten für das aktuelle Gerät auswählen
      device_data = df[df['device'] == device]
    
      # Normalisierung der Zeitdaten, mit Verschiebung, um mehr Farbsättigung zu erreichen
      time_normalized = (device_data['time'] - device_data['time'].min()) / (device_data['time'].max() - device_data['time'].min())
      time_normalized = time_normalized * 0.7 + 0.3  # Begrenzung auf den Bereich 0.3 bis 1.0
    
      # Punkte plotten und die Farbe entsprechend dem Zeitverlauf skalieren
      ax.scatter(device_data['x'], device_data['y'], device_data['z'], c=time_normalized, cmap=colors[i % len(colors)], s=point_size, label=device, marker='o')
    
      # Puntke laden für das aktuelle Gerät
      device_points = df[df['device'] == device][['x', 'y', 'z']].values
      # Schritt 1: Ebene der Punktwolke bestimmen
      center = np.mean(device_points, axis=0)  # Grobe Schätzung des Mittelpunkts
      pca = PCA(n_components=2)
      pca.fit(device_points - center)
      normal_vector = np.cross(pca.components_[0], pca.components_[1])  # Normalvektor der Ebene
    
      # Punktwolke in die Ebene projizieren
      plane_points = np.dot(device_points - center, pca.components_.T)  # Punkte in 2D-Ebene

      # Schritt 2: Kreis in 2D-Fit
      def residuals(params, x, y):
        xc, yc, r = params
        return np.sqrt((x - xc)**2 + (y - yc)**2) - r
    
      x_2d, y_2d = plane_points[:, 0], plane_points[:, 1]
      initial_guess = [0, 0, np.mean(np.sqrt(x_2d**2 + y_2d**2))]
      result = least_squares(residuals, initial_guess, args=(x_2d, y_2d))
      xc_2d, yc_2d, radius = result.x

      # Schritt 3: Kreis zurück in 3D transformieren
      v1, v2 = pca.components_
      center_circle_3d = center + xc_2d * v1 + yc_2d * v2
      circle_points = [
        center_circle_3d + radius * (np.cos(t) * v1 + np.sin(t) * v2)
        for t in np.linspace(0, 2 * np.pi, 100)
      ]
      circle_points = np.array(circle_points)

      # Kreis plotten
      ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color=colors2[i % len(colors)], label="Kreis_" + device)

      radi.append(radius)
      normal_vectors.append(normal_vector)
      circle_centers.append(center_circle_3d)

      #print("device: " + str(i) + device + " radius: " + str(radius) + " normalVektor: " + str(normal_vector))
      #print("centre: " + str(center_circle_3d))

    angle = math.degrees(calcAngle(normal_vectors[0], normal_vectors[1]))
    new_data = [
      {"file": filename, 
       "radius_object": radi[1],
       "radius_device": radi[0],
       "angle": angle,
       "centre_diff": np.linalg.norm(circle_centers[0] - circle_centers[1]),
       "centre_diff_x": circle_centers[0][0] - circle_centers[1][0],
       "centre_diff_y": circle_centers[0][1] - circle_centers[1][1],
       "centre_diff_z": circle_centers[0][2] - circle_centers[1][2],
       "x_normal_device": normal_vectors[0][0], 
       "y_normal_device": normal_vectors[0][1], 
       "z_normal_device": normal_vectors[0][2], 
       "x_normal_object": normal_vectors[1][0], 
       "y_normal_object": normal_vectors[1][1], 
       "z_normal_object": normal_vectors[1][2]
      },
    ]
    #print(str(new_data))
    append_to_csv(new_data)

# Achsenbeschriftungen
ax.set_xlabel("X-Achse")
ax.set_ylabel("Y-Achse")
ax.set_zlabel("Z-Achse")
ax.legend()  # Legende für die Geräte anzeigen
plt.show()

