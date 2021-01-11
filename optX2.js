class optX{
    constructor(){
        this.seeders=[];//[{config:[],loss:undefined,delta:undefined}];
        this.weights=[];
        this.model=null;
        this.modelLoss=undefined;
        this.index=-1;
        this.learningRate=1;
        this.best=-1;
    }

    newSeeder(args){
        return {config:[this.newSeederConfig(args)],loss:undefined,delta:undefined};
    }

    makeMutatedSeeder(seederIndex=0,level=0){
        
        var seeder=Object.assign({},this.seeders[seederIndex]);
        seeder.delta=seeder.loss=undefined;
        if(level>seeder.config.length)
            level=seeder.config.length;

        if(!level)
                seeder.config[0]=this.mutateSeederConfig(  seeder.config[0]);
        else{
                if(seeder.config.length===level)
                    seeder.config.push({mean:(10**-level)*seeder.config[level-1].mean});
                else
                    seeder.config[level]=
                        this.mutateSeederConfig( seeder.config[1],{
                            scale:10**-(level+2),
                            bias:10**-(level+2)
                        });
        }
        return seeder;
    }

    makeScaledSeeder(seederIndex=0,scale_=.5){
        var seeder=Object.assign({},this.seeders[seederIndex]);
        seeder.delta=seeder.loss=undefined;
        for(var i=0;i<seeder.config.length;i++)
            seeder.config[i].scale*=scale_;
        return seeder;
    }


    newSeederConfig(args){
        args=args||{};
        return {
            func:tf.randomNormal,
            seed:(Math.random()*2-1)*(args.seed||100),
            scale:args.scale||1,
            bias:args.bias||0,//need to be random
            mean:args.mean||0,
            deviation:args.deviation||1,
            next:args.next||1,
        }
    }


    mutateSeederConfig(config,args){
        args=args||{};
        args.scale=this._def(args.scale,.1)*(Math.random()*2-1);
        args.bias=this._def(args.bias,.1)*(Math.random()*2-1);

        var res= Object.assign({},config);
        //seeder.delta=seeder.loss=undefined;
        res.scale+=args.scale;
        res.bias+=args.bias;
        return res;
    }


    useModel(model){tf.tidy(()=>{
        this.model=model;
        if(this.weights && this.weights.length>0){
            //clean up
            for(var i=0;i<this.weights.length;i++)
            this.weights[i].dispose();
        }
        this.weights=[];var w=model.trainableWeights;
        for(var i=0;i<w.length;i++)
            this.weights.push(w[i].read().clone());
        this.modelLoss=undefined;
        return this.weights;
    });}

    set(seederIndex=-1){
        if(!this.model){
            console.error("useModel is not used to initialize.");
            return;
        }

        this.index=seederIndex;
        if (seederIndex==-1){
            for(var i=0;i<this.model.trainableWeights.length;i++)
                this.model.trainableWeights[i].write(this.weights[i]);
            return;
        }

        //this.model.trainableWeights=this.weights.concat();
        var l=this.weights.length;
        var w;
        var seeder=this.seeders[seederIndex];
        for(var i=0;i<l;i++){
            var w=this.weights[i];
            w=tf.tidy(()=>{
                for (var j=0;j<seeder.config.length;j++){
                    var s=seeder.config[j];
                    w=w.add(s.func(w.shape,s.mean,s.deviation,'float32',s.seed+s.next*i));
                }
                return w;
            });
            this.model.trainableWeights[i].write(w);
            w.dispose();
        }

    }

    add(args){
        if(!args)args={scale:this.learningRate};
        this.seeders.push(this.newSeeder(args));
    }

    addMutate(seederIndex=0,level=0){
        this.seeders.push(this.makeMutatedSeeder(seederIndex,level));
    }

    addMultiply(seederIndex=0,val=.5){
        this.seeders.push(this.makeScaledSeeder(seederIndex,val));
    }

    evaluate(ins,outs,args){
        var i=-2;
        this.best=-1;
        var waitObject={
            i:0,
            onReady:()=>{console.log("onReady not set")},
            best:-1,
            bestLoss:undefined,
        };
        while(++i<this.seeders.length){
            this.set(i);
            waitObject.i++;
            var _e=this.model.evaluate(ins,outs,args);
            _e=(Array.isArray(_e)?_e[0]:_e);
            this.setLoss(i,_e,waitObject);
        }

        return waitObject;
    }

    setLoss(seederIndex,tensor,waitObject){
        tensor.data().then((d)=>{
            if(seederIndex===-1)
                this.modelLoss=d[0];
            else{
                this.seeders[seederIndex].loss=d[0];
                this.seeders[seederIndex].delta=d[0]-this.modelLoss;
            }

            if(waitObject){
                waitObject.i--;
                if(waitObject.loss===undefined || waitObject.loss>d[0]){
                    waitObject.best=seederIndex;
                    waitObject.loss=d[0];
                }

                if(waitObject.i<1){
                    this.set(waitObject.best);
                    waitObject.onReady();
                }
            }
        });
    }

    

    sortSeeders(){
        if(this.modelLoss===undefined){
            console.warn("Unable to sort Seeders");
            return;
        }
        this.seeders.sort((a,b)=>a.loss>b.loss?1:-1);
    }

    cleanSeeders(max=0,noException=false){
        this.sortSeeders();
        if(max===0)
            max=this.seeders.length-2;
        else
            max=this.seeders.length-max;
        while(max-->0)
            if(this.seeders[this.seeders.length-1].delta>0 || noException)
                this.seeders.pop();
    }

    _def(src,def_){
        return src!==undefined?src:def_;
    }
}