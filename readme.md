[seamk_logo]:       /img/Seamk_logo.svg
[epliitto_logo]:    /img/EPLiitto_logo_vaaka_vari.jpg

# Datan luonti ja GAN 

Tekoäly-AKKE hankkeessa yhtenä demona kokeiltiin synteettisen datan luontia mittaustulosten perusteella. Jos käytössä ei ole riittävästi dataa tekoälymallin opettamiseen, voi synteettinen data avata mahdollisuuden luoda näennäisesti oikean mallista dataa millä voi täydentää mittaustuloksia. 

Tässä demossa on syöttönä on ulkolämpötilan mittauksia viiden minuutin välein toukokuusta 2021 heinäkuulle 2022. 

## Luo ajoympäristö

Ohjelma on kirjoitettu pythonilla, joten oletamme alla, että se on jo asennettuna. Osa kirjastoista ei toimi tätä kirjoitettaessa uusimmilla Pythonin versioilla. Mikäli kohtaa ongelmia, ainakin versiolla 3.8.13 toimii. [pyenv](https://github.com/pyenv/pyenv)  auttaa useamman Python version kanssa elämistä.

Kun olet kloonannut repositoryn johonkin hakemistoon, luo ympäristö sinne 

```
python -m venv venv
```

Aktivoi ympäristö linux/mac `source venv/bin/activate`, windows powershell `.\venv\Scripts\Activate.ps1`. Tämän jälkeen voi asentaa tarvittavat kirjastot tähän virtuaaliympäristöön. 

```
pip install --upgrade pip wheel
pip install Cython --install-option="--no-cython-compile"
pip install --upgrade pandas numpy ctgan
```

Viimeinen asennus saattaa kestää tovin. 

## Aja ohjelma

```
python temperature_gan_esimerkki.py
```

Ohjelma lukee data tiedoston sisään, käsittelee mallia, opettaa GAN verkon luomaan vastaavan laista dataa ja tulostaa lopuksi ulos kuinka lähelle luotu data vastaa mitattua. 


## Tekoäly-AKKE hanke

Syksystä 2021 syksyyn 2022 kestävässä hankkeessa selvitetään Etelä-Pohjanmaan alueen yritysten tietoja ja tarpeita tekoälyn käytöstä sekä tuodaan esille tapoja sen käyttämiseen eri tapauksissa, innostaen laajempaa käyttöä tällä uudelle teknologialle alueella. Hanketta on rahoittanut Etelä-Pohjanmaan liitto.

![epliitto_logo]

---

![seamk_logo]
