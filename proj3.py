import cv2  # Az OpenCV könyvtár importálása

# A rendszám detektáláshoz szükséges Haar-kaskád betöltése
plateCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")  # Rendszám felismeréshez szükséges kaskád fájl betöltése

# Kép betöltése
img = cv2.imread("p4.jpg")  # A 'p1.jpg' képet betöltjük

# Kép átalakítása szürkeárnyalatossá
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # A képet szürkeárnyalatosra alakítjuk, mivel a Haar-kaskád szürkeárnyalatú képeken működik

# Rendszámok detektálása
plates = plateCascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
# A detectMultiScale függvény a képen található rendszámokat próbálja detektálni
# scaleFactor: Kép méretének csökkentési aránya a vizsgálat során
# minNeighbors: A rendszámokat elfogadó objektumok számára minimális szomszédsági követelmény
# minSize: A detektálandó rendszámok minimális mérete

# A detektált rendszámok kiemelése
for (x, y, w, h) in plates:  # Az 'plates' lista minden egyes elemére (a rendszámok koordinátái)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # A rendszámok köré téglalapot rajzolunk (kék színnel, 2-es vastagsággal)

# Az eredmény megjelenítése
cv2.imshow("rendszam", img)  # Az eredeti képet a detektált rendszámokkal együtt jelenítjük meg

# A program vár egy billentyű lenyomására, mielőtt bezárja az ablakot
cv2.waitKey(0)  # Az ablak addig nyitva marad, amíg nem nyomunk meg egy billentyűt

# Minden ablak bezárása
cv2.destroyAllWindows()  # Az összes megnyitott OpenCV ablak bezárása
