   Info<< "Reading field T\n" << endl;

        Info<< "Reading field q\n" << endl;
    volScalarField q
    (
        IOobject
        (
            "q",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );


    volScalarField T
    (
        IOobject
        (
            "T",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );
//----------------------------------------
    Info<< "Reading field p_rgh\n" << endl;
    volScalarField p_rgh
    (
        IOobject
        (
            "p_rgh",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );
//------------------------------------------
    Info<< "Reading field U\n" << endl;
    volVectorField U
    (
        IOobject
        (
            "U",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );
//---------------------------------------------
    #include "createPhi.H"

    #include "readTransportProperties.H"
//--------------------------------------------
    Info<< "Creating turbulence model\n" << endl;
    autoPtr<incompressible::RASModel> turbulence    // this 3 lines are different from LakebuoyantBussinesqPimpleFoam
    (
        incompressible::RASModel::New(U, phi, laminarTransport)
    );
//--------------------------------------------
    // Kinematic density for buoyancy force
    Info<< "Reading field rhok\n" << endl;
    volScalarField rhok
    (
        IOobject
        (
            "rhok",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        1.0 - beta*(T - TRef)
    );
//-----------------------------------------------
    // kinematic turbulent thermal thermal conductivity m2/s
    Info<< "Reading field alphat\n" << endl;
    volScalarField alphat
    (
        IOobject
        (
            "alphat",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );
//----------------------------------------------
/*    Info<< "Calculating field zpos\n" << endl;
    volScalarField zpos
    (
        IOobject
        (
            "zpos",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zpos",dimensionSet(0,1,0,0,0),0.0)
      );   
    zpos=mesh.C().component(2);
*/
    Info<< "Calculating field g.h\n" << endl;
    volScalarField gh("gh", g & mesh.C());
    surfaceScalarField ghf("ghf", g & mesh.Cf());
/*
//---------------------------------------------------------
     const dimensionedScalar zmax=max(mesh.Cf().component(2));
//---------------------------------------------------------
     Info<< "Calculating field zStarpos\n" << endl;
     volScalarField zStarpos
     (
        IOobject
        (
            "zStarpos", // zStarpos is the depth of point from the water surface
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zStarpos",dimensionSet(0,1,0,0,0),0.00000001)
      );   
    zStarpos=zmax-zpos;
//------------------------------------------------------------
    Info<< "Calculating field sourceT\n" << endl;
    expressionSource<scalar> sourceTerm
    (
        IOdictionary
        (
            IOobject
            (
                "sourceTermDict",
                runTime.constant(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            )
        ).subDict("TEqn"),
        mesh
    );
//------------------------------------------------------
*/
    volScalarField p
    (
        IOobject
        (
            "p",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        p_rgh + rhok*gh
     );
//---------------------------------------------------
    label pRefCell = 0;
    scalar pRefValue = 0.0;
    setRefCell
    (
        p,
        p_rgh,
        mesh.solutionDict().subDict("PIMPLE"),
        pRefCell,
        pRefValue
    );

    if (p_rgh.needReference())
    {
        p += dimensionedScalar
        (
            "p",
            p.dimensions(),
            pRefValue - getRefCellValue(p, pRefCell)
        );
    }

