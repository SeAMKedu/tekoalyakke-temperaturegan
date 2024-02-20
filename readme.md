[seamk_logo]:       /img/Seamk_logo.svg
[epliitto_logo]:    /img/EPLiitto_logo_vaaka_vari.jpg


[doy1]:             /img/plot-doy-0.01.png
[dom1]:             /img/plot-dom-0.01.png
[di1]:              /img/plot-di-0.01.png
[doy2]:             /img/plot-doy-0.05.png
[dom2]:             /img/plot-dom-0.05.png
[di2]:              /img/plot-di-0.05.png
[doy3]:             /img/plot-doy-0.1.png
[dom3]:             /img/plot-dom-0.1.png
[di3]:              /img/plot-di-0.1.png
[doy4]:             /img/plot-doy-0.25.png
[dom4]:             /img/plot-dom-0.25.png
[di4]:              /img/plot-di-0.25.png
[doy5]:             /img/plot-doy-0.5.png
[dom5]:             /img/plot-dom-0.5.png
[di5]:              /img/plot-di-0.5.png
[doy6]:             /img/plot-doy-0.75.png
[dom6]:             /img/plot-dom-0.75.png
[di6]:              /img/plot-di-0.75.png
[doy7]:             /img/plot-doy-1.0.png
[dom7]:             /img/plot-dom-1.0.png
[di7]:              /img/plot-di-1.0.png

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10682576.svg)](https://doi.org/10.5281/zenodo.10682576)

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
pip install --upgrade pandas numpy sdv matplotlib seaborn
```

Viimeinen asennus saattaa kestää tovin. 

## Esimerkki 1

```
python temp_gen_sample1.py
```

Ohjelma lukee data tiedoston sisään, käsittelee mallia, opettaa GAN verkon luomaan vastaavan laista dataa ja tulostaa lopuksi kuinka lähelle luotu data vastaa mitattua. 

```
Sampled data comparison:

Time:  2021-05-02 19:20:01 - 2021-05-02 19:25:01  True value:  1.9  Generated value:  1.8
Time:  2021-05-10 04:00:01 - 2021-05-10 04:05:01  True value:  5.6  Generated value:  4.1
Time:  2021-05-09 13:45:01 - 2021-05-09 13:50:01  True value:  12.7  Generated value:  8.4
Time:  2021-05-03 15:45:01 - 2021-05-03 15:50:01  True value:  8.9  Generated value:  4.2
```

## Esimerkki 2 

```
python temp_gen_sample2.py
```

Tämä esimerkki lukee saman datan sisään, mutta ajaa syntetisoinnin useilla eri data määrillä sekä koittaen eri tapoja datan mallintamisessa (vuoden päivän mukaan. doy; vuosi, kuukausi, päivä eroteltuna, dom; indeksin mukaan, di). 

Alle olevat kuvat näyttävät kuinka opettamisessa käytetyn datan lisääminen parantaa tulosta, mutta myös kuinka se lisää tarvittavaa aikaa. Kuvista näkyy myös kuin tietyt tavat mallintaa ongelmaa vaikuttaisivat soveltuvan paremmin kuin toiset. 

| frac | doy     | dom     | di     | rows   | time |
|  --: | :--:    | :--:    | :--:   | --:    | --:  |
| 0.01 | ![doy1] | ![dom1] | ![di1] | 1300   | 15s  |
| 0.05 | ![doy2] | ![dom2] | ![di2] | 6500   | 50s  |
| 0.10 | ![doy3] | ![dom3] | ![di3] | 13000  | 100s |
| 0.25 | ![doy4] | ![dom4] | ![di4] | 32500  |  4m  |
| 0.50 | ![doy5] | ![dom5] | ![di5] | 65000  |  8m  |
| 0.75 | ![doy6] | ![dom6] | ![di6] | 97500  | 12m  |
| 1.00 | ![doy7] | ![dom7] | ![di7] | 130000 | 15m  |

Keskimääräinen virhe hieman vaikuttaa parantuvan datan lisääntyessä, mutta ei merkittävästi. Tätä voi parantaa hakemalla erilaisia käytettyjen tasojen määriä ja muiden hyperparametrien säätämisellä.

| Fraction | di Error | dom error | doy error |
| --: | --: | --: | --: |
| 0.01  | 16.607000 | 10.955208 | 12.408000 |
| 0.05  | 11.193000 | 12.068000 | 12.529000 |
| 0.10  | 11.075000 | 11.269000 | 16.727000 |
| 0.25  | 13.639000 | 10.548980 | 10.772000 |
| 0.50  | 11.864000 | 11.102083 | 9.468000  |
| 0.75  | 11.748000 | 11.637234 | 10.208000 |
| 1.00  | 13.637000 | 12.716000 | 10.680000 |


## Tekoäly-AKKE hanke

Syksystä 2021 syksyyn 2022 kestävässä hankkeessa selvitetään Etelä-Pohjanmaan alueen yritysten tietoja ja tarpeita tekoälyn käytöstä sekä tuodaan esille tapoja sen käyttämiseen eri tapauksissa, innostaen laajempaa käyttöä tällä uudelle teknologialle alueella. Hanketta on rahoittanut Etelä-Pohjanmaan liitto.

![epliitto_logo]

---

![seamk_logo]
