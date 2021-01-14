if(!window._guzuTF)window._guzuTF={};
window._guzuTF.optX=class optX{
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
        
        var seeder=this._parse(this.seeders[seederIndex]);
        seeder.delta=seeder.loss=undefined;
        if(level>seeder.config.length)
            level=seeder.config.length;

        if(!level)
                seeder.config[0]=this.mutateSeederConfig(  seeder.config[0]);
        else{
                if(seeder.config.length===level)
                    seeder.config.push(this.newSeederConfig({
                        mean:(10**-level) * (seeder.config[level-1].mean),
                        scale:(10**-level)* (seeder.config[level-1].scale)
                    }));
                else
                    seeder.config[level]=
                        this.mutateSeederConfig( seeder.config[level],{
                            scale:10**-(level+1),
                            bias:10**-(level+1)
                        });
        }
        //no need for smaller levels if top modified
        while(level<seeder.config.length-1)
            seeder.config.pop();

        return seeder;
    }

    _parse(x){
        return JSON.parse(JSON.stringify(x));
    }
    makeScaledSeeder(seederIndex=0,scale_=.5,shift=0){
        var seeder=this._parse(this.seeders[seederIndex]);
        seeder.delta=seeder.loss=undefined;
        for(var i=0;i<seeder.config.length;i++){
            var _00=seeder.config[i].scale;
            seeder.config[i].scale*=scale_;
            if(!Number.isFinite(seeder.config[i].scale))
                console.log("Infinit...")
        }
        seeder.config[0].bias+=shift;
        return seeder;
    }


    newSeederConfig(args){
        args=args||{};
        return {
            func:tf.randomNormal,
            seed:(Math.random()*2-1)*(args.seed||100),
            scale:args.scale||1,
            bias:args.bias||0,//need to be random
            mean:args.mean||(Math.random()*(args.scale||0)*0.01),
            deviation:args.deviation||1,
            next:args.next||1,
            pow:args.pow||(Math.floor(Math.random()*3)*2+1),
            notAll:Math.random()<.5?undefined:(Math.random()*2-1)*.9,//some weights won't be modified
        }
    }


    mutateSeederConfig(config,args){
        args=args||{};
        args.scale=this._def(args.scale,.01)*(Math.random()*2-1);
        args.bias=this._def(args.bias,.01)*(Math.random()*2-1);

        var res= this._parse(config);
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
        var sfunc=tf.randomNormal;
        for(var i=0;i<l;i++){
            var w=this.weights[i];
            w=tf.tidy(()=>{
                for (var j=0;j<seeder.config.length;j++){
                    var s=seeder.config[j];

                    w=w.add(sfunc(w.shape,s.mean,s.deviation,'float32',s.seed+s.next*i)).pow(s.pow).mul(s.scale).add(s.bias);
                    //document.getElementById("1").innerHTML=("scale: "+s.scale+"<br>bias: "+s.bias)+"<br>";
                    if(!Number.isFinite(s.scale))
                        console.log("INFINITY!");
                    if(s.notAll)
                        w=w.mul(sfunc(w.shape,1,1,'float32',-(s.seed+s.next*i)-10.99).greater(tf.zerosLike(w).sub(s.notAll)));
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

    addMultiply(seederIndex=0,val=.5,shift=0){
        this.seeders.push(this.makeScaledSeeder(seederIndex,val,shift));
    }

    evaluate(ins,outs,args){
        var i=-2;
        this.best=-1;
        var waitObject={
            i:0,
            onReady:()=>{console.warn("onReady not set")},
            best:-1,
            bestLoss:undefined,
        };
        while(++i<this.seeders.length){
            this.set(i);
            waitObject.i++;
            var _e;
            _e=this.model.evaluate(ins,outs,args);
            this.yo=_e;
            
            //_e=(Array.isArray(_e)?_e[0]:_e);
            this.setLoss(i,(Array.isArray(_e)?_e[0]:_e)
                ,waitObject);
        }

        return waitObject;
    }

    setLoss(seederIndex,tensor,waitObject){
        tensor.data().then((d)=>{
            //if(isNaN(d[0]))console.log("Something wrong.")

            if(seederIndex===-1)
                this.modelLoss=(
                    this.modelLoss===undefined?0:this.modelLoss
                    )+d[0];

            else{
                var u=this.seeders[seederIndex].loss;
                u=u===undefined?0:u;
                u=this.seeders[seederIndex].loss=u+d[0];
                this.seeders[seederIndex].delta=u-this.modelLoss;
                
            }

            if(waitObject){
                waitObject.i--;
                if(waitObject.loss===undefined || waitObject.loss>d[0]){
                    waitObject.best=this.best=seederIndex;
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
        this.seeders.sort((a,b)=>a.loss>b.loss ||isNaN(a.loss) && !isNaN(b.loss)?1:-1);
    }

    cleanSeeders(max=0,noException=false){
        if (!this.seeders.length){
            console.log("Nothing to clean");
            return;
        }
        this.sortSeeders();
        if(max===0)
            max=this.seeders.length-2;
        else
            //if(!noException)
          max=this.seeders.length-max;
        while(max-->0)
            if(this.seeders[this.seeders.length-1].delta>0 || noException)
                this.seeders.pop();
        if(this.best!=-1)this.best=0;
    }

    removeLoss(){
        //this.modelLoss=undefined;
        var i=-1;while(++i<this.seeders.length)
            this.seeders[i].loss=this.seeders[i].delta=undefined;
    }

    _def(src,def_){
        return src!==undefined?src:def_;
    }

    getLoss(){
        return this.best==-1?this.modelLoss:
            this.seeders[this.best].loss;
    }
}
//optX






//
class optX2{
    constructor(model,args){
        args=args||{};
        this.searchSize=args.searchSize||20;
        this._instate={};

        this.retries=args.retries||3;//retry if failed

        this._retries=0;

        this.model=model;
        this.optX=new window._guzuTF.optX();
        this.optX.learningRate=args.learningRate||1;

        this.state="search";//search,success,failure
        this._stageCounter=0;
        this.isTraining=false;
        this.stopTraining=false;
    }

    fit(inputs,outputs,args){
        args=args||{};
        this._instate={};//new instate
        this._fit([inputs,outputs,args],args.epochs);
        this.isTraining=true;
        args.callbacks.then=(f)=>args.callbacks.onTrainEnd=f;
        return args.callbacks;
    }

    async _fit(args,epochs){
        this.optX.useModel(this.model);
        this.optX.evaluate(args[0],args[1],args[2]).onReady=(()=>{
            //console.log("obest"+this.optX.best)
            if(this.stopTraining ||!epochs){
                this.stopTraining=this.isTraining=false;
                return;
            }
            var loss=this.optX.getLoss();
            if(--epochs>0){
                this._nextGeneration();
                this._fit(args,epochs);
            }
            else
                this.isTraining=false;
            var cb=args[2].callbacks;
            if(cb){
                if(cb.onEpochEnd)
                    cb.onEpochEnd(epochs+1,{loss:loss});
                if(epochs===0 && cb.onTrainEnd)
                    cb.onTrainEnd();
            }
        });
    }



    _nextGeneration(){
        var i;
        var o=this.optX;
        
        if(o.best!=-1){
            console.log("Success after "+this.state)
            this.state="success";
            this._instate.searchTries=1;
            this._retries=0;
        }
        else
        if(this._retries++<this.retries && this.state!="search")
            this.state="failure"
        else{
            this.state="search";
            this._retries=0;
        }

        switch(this.state){
            case "search":
                o.seeders=[];//throw all
                var st=this._instate.searchTries=(this._instate.searchTries||0)+1;
                if(st>5)st=1;
                i=-1;while(++i<this.searchSize){
                    o.add({
                        //half LR, half smaller LR
                        scale:i<this.searchSize*.5?
                            o.learningRate:
                            o.learningRate*(10**-st)
                    });
                    o.add({
                        //half LR, half bigger LR
                        scale:i<this.searchSize*.5?
                            o.learningRate:
                            o.learningRate*(10**st)
                    });
                }
                
                break;
            case "success":
                console.log("success")
                //keep some
                var ss=Math.ceil(this.searchSize*.3)||1;
                o.cleanSeeders(0);
                o.cleanSeeders(ss,true);
                o.removeLoss();
                //
                o.addMultiply(0,.5);
                o.addMultiply(0,2);
                
                i=0;
                while(o.seeders.length<this.searchSize)
                    o.addMutate((i++)%ss,i%3);

                break;
            case "failure":
                //keep some
                //var ss=Math.ceil(this.searchSize*.05)||1;
                //console.log("index:"+o.index)
                o.sortSeeders();o.seeders=[o.seeders[0]];
                console.log(this._retries+" least bad: "+o.seeders[0].loss+"\t\t\tmodLoss:"+o.modelLoss);
                //epic fail
                if(isNaN(o.seeders[0].loss) || o.seeders[0].loss>1e10){
                    console.log("EPIC FAIL");
                    this.state="search";
                    return this._nextGeneration();
                }

                o.removeLoss();
                var i=0;
                //var ss=Math.ceil(this.searchSize*.1)||1;
                //o.seeders=o.seeders.slice(0,ss);
                
                while(o.seeders.length<this.searchSize){
                    o.addMutate(0,i%5);
                    i++;
                }
                

                break;
        }

        //console.log(this.state);

        
    }

}