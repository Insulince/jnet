package main

import (
	"fmt"
	"github.com/Insulince/jnet"
	"log"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().Unix())
}

func main() {
	nw, err := jnet.Deserialize(`1.0
4 4 4 8 4
2.672999788482761 -2.7088689357170344 2.8604848562446565 -0.04411667193355516
2.983835343405863 -2.6901750050370294 2.9531085947371367 -2.0913287476036557
1.6768757898256643 -0.050454644983101385 -0.3057881832017168 -1.2694241083868454 1.5776314456560616 -0.484253425030897 0.8808382342834659 -2.2998275057230186
-4.383896269695666 0.8387218593054648 -0.08085282941772509 -2.8400332449457792
-3.6608097468450556 -0.14178913255625447 -2.380728662087128 0.751223138926559
1.8844157950955165 -5.152771389222051 4.2212608555935445 -2.937490555106668
-5.232232158067436 -2.5869271408376258 -2.4571827543438762 -5.215132502445575
-1.7366738486690203 2.8994680199851013 -4.476142960009187 -0.38691798911592795
-2.9029732601393623 -5.6438551617477275 5.081808455841638 -3.0447055792751874
2.133763589265869 5.138386368024666 -3.4287357719903637 2.850822844057835
-2.395753852109854 -5.549135891063204 4.698763160257634 -3.2717301211853775
4.810610820316556 4.959100492266656 -7.416339582657778 4.373464032899299
4.98203868274197 -4.663407468371152 4.772578128609006 -1.2411242000967875
0.31826699970724215 -0.6821802163652814 1.1354066791132817 0.5509391796619202
-4.881988989874827 3.79291725031201 -4.351409890697336 5.27566507749001
1.8444038596795114 -0.8153632625066625 2.631551380840467 -5.154280558889697
5.066297502895604 -3.6622024240498874 5.609481963952589 -2.1906193808678034
5.958207097815084 -3.4138070312364914 4.746272205116761 -6.880330842380986
0.8734136941978646 1.701404528879978 0.17753269587552373 0.761563940202219
3.6861140661542815 -4.8302617766996665 4.116705486222697 -9.5280944866097
-2.653808291540114 -1.6475673563934814 -5.612999123819763 3.5436661009879726 -0.16435972486527112 3.8180868741245733 -2.30022662121479 9.965781020669217
-6.004485717439309 -1.5360997843578943 5.850258679744891 -1.5364951619924498 -6.043463364663935 -4.02819304992811 0.7473702499738913 -4.254216821089913
1.8960364475430511 -0.2460287844938935 -11.337971329200858 -3.24632005837162 3.3096473156070667 3.686683429646359 -1.2347745930929468 -11.000907282233165
6.640146213785366 -0.46568764654906497 -0.07381945943029312 -4.978154365964497 6.7447227201646935 -12.03713713040157 -3.5241877990923944 -7.012703492737283
solid vertical diagonal horizontal`)
	if err != nil {
		log.Fatalln(err)
	}

	input := []float64{
		1, 0,
		0, 1,
	}
	prediction, err := nw.Predict(input)
	if err != nil {
		log.Fatalln(err)
	}
	fmt.Println(prediction)
}