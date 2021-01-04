class optX{
    constructor(){
        this.seeds=[{seed:undefined,loss:undefined,delta:undefined}];
        this.weights=[];
        this.model=null;
        this.best=0;
    }

    init(model){
        if(this._evaluating)
            return console.warn("init while still evaluating.")
        this.model=model;
        //if(this.weights.length>0)
        //    for(var i in this.weights)
        //        this.weights[i].dispose();    
        this.weights=model.getWeights().concat();
        this.weights0=this._cloneWeights(this.weights);
        //for(var i in this.weights)
        //    this.weights[i]=this.weights[i].clone();
    };

    _cloneWeights(w){
        var r=[];
        for(var i=0;i<w.length;i++)
            r[i]=w[i].clone();
        return r;
    }

    add(seed){
        if(this.seeds.length===0){
            this.seeds.push({seed:null,loss:undefined});
            return;
        }
        seed=(this.repairSeed(seed));
        var m={
            seed:seed,
            loss:1,
        };

        this.seeds.push(m);
    }

    addMultiply(index,val){
        if(!index)return;//don't add for zero
        var m={
            seed:this.seeds[index].seed.concat(),
            loss:1,
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
            loss:1,
        };
        m.seed[5]=this.repairSeed(m.seed[5][2]);//new sub-seed
        m.seed[5].pop();//internal seed not needed
        this.seeds.push(m); 
    }


    getLength(){
        return this.seeds.length;
    }
    
    get(index){
        if(index===0)return this.weights.concat();
        return this.getMutation(this.weights,this.seeds[index].seed);
    }

    set(model,seedIndex){
        if(!seedIndex)
            model.setWeights(this.weights0);
        else
            this.setMutation(model,this.seeds[seedIndex].seed);
    }

    evaluate(ins,outs,args){
        this.best=0;this._evaluating=true;
        var i=-1;
        
        var ev=()=>{ 
            var m=this.model;
            i++;if(i<this.seeds.length){
               
                //tf.tidy(()=>
                if(i>0)
                    m.setWeights(this.get(i));
                else
                    this.weights=this._cloneWeights(m.getWeights());
                //);
                var _e=m.evaluate(ins,outs,args);
                (Array.isArray(_e)?_e[0]:_e).data().then((d)=>{
                    this.seeds[i].loss=d[0];
                    this.seeds[i].delta=0;
                    if(i>0)this.seeds[i].delta=d[0]-this.seeds[0].loss;
                    if(this.seeds[i].delta<this.seeds[this.best].delta){
                        this.best=i;
                        //m.setWeights(this.weights)
                    }
                   ev();
                });
            }else{
                this._evaluating=false;
                m.setWeights(this.weights0.concat());
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
        while(max-->0)
            if(this.seeds[this.seeds.length-1].delta>0 || noException)
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
        if(!seed)
            return this._cloneWeights(w);//w.concat();
        w=w.concat();
        for (var i in w){
            //major seed applied
            w[i]=w[i].add(
               seed[3](w[i].shape,seed[0],seed[1],'float32',seed[2]).mul(seed[4])
               );
            //minor seed applied
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