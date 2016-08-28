class MatlabRP : public MatlabBP 
{
    protected:
        REAL deltaMin;
        REAL deltaMax;
        REAL initEta;
        REAL incEta;
        REAL decEta;
    
    public:
        MatlabRP(const mxArray *netStr, const mxArray *trnParam) : MatlabBP(netStr, trnParam)
        {
            DEBUG1("Initializing the RProp class from a Matlab Network structure.");
            if (mxGetField(trnParam, 0, "deltamax")) this->deltaMax = static_cast<REAL>(mxGetScalar(mxGetField(trnParam, 0, "deltamax")));
            else this->deltaMax = 50.0;
            if (mxGetField(trnParam, 0, "min_grad")) this->deltaMin = static_cast<REAL>(mxGetScalar(mxGetField(trnParam, 0, "min_grad")));
            else this->deltaMin = 1E-6;
            if (mxGetField(trnParam, 0, "delt_inc")) this->incEta = static_cast<REAL>(mxGetScalar(mxGetField(trnParam, 0, "delt_inc")));
            else this->incEta = 1.10;
            if (mxGetField(trnParam, 0, "delt_dec")) this->decEta = static_cast<REAL>(mxGetScalar(mxGetField(trnParam, 0, "delt_dec")));
            else this->decEta = 0.5;
            if (mxGetField(trnParam, 0, "delta0")) this->initEta = static_cast<REAL>(mxGetScalar(mxGetField(trnParam, 0, "delta0")));
            else this->initEta = 0.1;
        }
    
        virtual Backpropagation *getNetwork()
        {
            RProp *ret = new RProp(numNodes, trfFunc, usingBias, deltaMin, deltaMax, initEta, incEta, decEta);
            ret->readWeights( (const REAL***) weights, (const REAL**) bias);
            for (list<Node>::const_iterator itr = frozen.begin(); itr != frozen.end(); itr++) ret->setFrozen(itr->layer, itr->node, true);
            return ret;
        }
};