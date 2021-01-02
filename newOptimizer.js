class optX{
    constructor(){
        this.seeds=[];
        this.weights=[];
        this.model=null;
        this.best=0;
    }

    init(model){
        if(this._evaluating)
            console.warn("init while still evaluating.")
        this.model=model;
        //if(this.weights.length>0)
        //    for(var i in this.weights)
        //        this.weights[i].dispose();    
        this.weights=model.getWeights();
        
        for(var i in this.weights)
            this.weights[i].clone();
    };

    add(seed){
        if(this.seeds.length===0){
            this.seeds.push({seed:null,loss:undefined});
            return;
        }
        seed=(this.repairSeed(seed));
        var m={
            seed:seed,
            loss:undefined,
        };

        this.seeds.push(m);
    }

    addMultiply(index,val){
        if(!index)return;//don't add for zero
        var m={
            seed:this.seeds[index].seed.concat(),
            loss:undefined,
        };
        val=val||1;
        m.seed[4]=m.seed[4]*val;
        this.seeds.push(m);
    }

    addMutate(index){
        if(!index)index=this.best;
        if(!index)return;//don't add for zero
        var m={
            seed:this.seeds[index].seed.concat(),
            loss:undefined,
        };
        m.seed[5]=this.repairSeed(m.seed[5][2]);//new sub-seed
        m.seed[5].pop();//internal seed not needed
        this.seeds.push(m); 
    }


    getLength(){
        return this.seeds.length;
    }
    
    get(index){
        if(index===0)return this.weights;
        return this.getMutation(this.weights,this.seeds[index].seed);
    }

    set(model,seedIndex){
        if(!seedIndex)
            model.setWeights(this.weights);
        else
            this.setMutation(model,this.seeds[seedIndex].seed);
    }

    evaluate(ins,outs,args){
        this.best=0;this._evaluating=true;
        var i=-1;
        
        var ev=()=>{
            i++;if(this.seeds.length>i){
                var m=this.model;
                //tf.tidy(()=>
                m.setWeights(this.get(i))
                //);
                m.evaluate(ins,outs,args).data().then((d)=>{
                    this.seeds[i].loss=d[0];
                    this.seeds[i].delta=0;
                    if(i>0)this.seeds[i].delta=d[0]-this.seeds[0].loss;
                    if(this.seeds[i].loss<this.seeds[this.best].loss){
                        this.best=i;
                        //m.setWeights(this.weights)
                    }
                   ev();
                });
            }else{
                this._evaluating=false;
                //console.log("done evaluation")
                if(this._then){
                    var then=this._then;this._then=undefined;
                    then();
                    
                }
            }
        };

       ev();
       return this;
    }

    then(f){
        this._then=f;
    }

    getBest(){
        return this.seeds[this.best];
    }

    sortSeeds(){
        var a=this.seeds.shift();
        this.seeds.sort((a,b)=>a.loss>b.loss?1:-1);
        this.seeds.unshift(a);
        if(this.best>0)this.best=1;
    };

    cleanSeeds(max=0,noException=false){
        this.sortSeeds();
        if(max===0)
            max=this.getLength()-2;
        else
            max=this.getLength()-max;
        var s=this.seeds;
        while(max-->0)
            if(s[s.length-1].delta>0 || noException)
                this.seeds.pop();
    }
    
    



    repairSeed(seed){
        //seed=[mean,davi,seed,type,multiply]
        if(!seed)seed=Math.random()*1000;
        if(!isNaN(seed))seed=[0,0.0001,seed,tf.randomNormal,1,[0,0.000001,seed,tf.randomNormal,1]];//make seed array
        return seed;
    }

    getMutation(w,seed){return tf.tidy(()=>{
        //var w=model.getWeights();
        
        //seed=this.repairSeed(seed);
        
        
        for (var i in w){
            //major seed
            w[i]=w[i].add(
               seed[3](w[i].shape,seed[0],seed[1],'float32',seed[2]).mul(seed[4])
               );
            //minor seed
            w[i]=w[i].add(
            seed[5][3](w[i].shape,seed[5][0],seed[5][1],'float32',seed[5][2]).mul(seed[5][4])
            );
        }

        
        return w;
    })};

    setMutation(model,seed){
        model.setWeights(!seed &&seed!=0?this.weights:
            this.getMutation(this.weights,seed));
    }

    
    



}