    if (runTime.outputTime())
    {
        //volVectorField gradux=fvc::grad(ux);
        //volVectorField graduy=fvc::grad(uy);        
        volVectorField gradT(fvc::grad(T));
        volTensorField gradU(fvc::grad(U));
        volScalarField gradTy
        (
            IOobject
            (
                "gradTz",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            gradT.component(vector::Z)
        );
        
/*       volScalarField gradUx
        (
            IOobject
            (
                "gradUx",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
             gradux.component(1)           
//gradU.component(vector::x)
        );
       volScalarField gradUz
        (
            IOobject
            (
                "gradUz",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
             graduy.component(0);
            //gradU.component(vector::z)
        );
*/ 
          volScalarField TSrc=sourceTerm();
           TSrc.rename("TSrc");
           TSrc.write();

        runTime.write();
    }

