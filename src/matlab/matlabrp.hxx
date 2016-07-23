      /// Constructor taking the parameters for a matlab net structure.
      /**
      This constructor should be called when the network parameters are stored in a matlab
      network structure.
      @param[in] netStr The Matlab network structure as returned by newff.
      @param[in] trnParam The Matlab network train parameter (net.trainParam) structure.
      */
  RProp::RProp(const mxArray *netStr, const mxArray *trnParam) : Backpropagation(netStr, trnParam)
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

    try {allocateSpace(nNodes);}
    catch (bad_alloc xa) {throw;}

    //Initializing the dynamically allocated values.
    for (unsigned i=0; i<(nNodes.size() - 1); i++)
    {
      for (unsigned j=0; j<nNodes[i+1]; j++) 
      {
        prev_db[i][j] = 0.;
        delta_b[i][j] = this->initEta;
        
        for (unsigned k=0; k<nNodes[i]; k++)
        {
          prev_dw[i][j][k] = 0.;
          delta_w[i][j][k] = this->initEta;
        }
      }
    }
  }
