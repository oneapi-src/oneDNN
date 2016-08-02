// This works in C

function1()
{
    engine = engine_create( ... );
    convolution = convolution_create( ... , engine, ... );
    return convolution;
}

function2(convolution)
{
    stream = stream_create();
    stream.submit(convolution);
}

function3(convolution, stream)
{
    stream.wait();
    engine = primitive_get_primitive_descriptor(convolution).engine;
    // or engine = primitive_get_engine(convolution)
    engine_destroy(engine);
    ...
}

// C++

function1() {
    engine = engine( ... );
    convolution = convolution( ... , engine, ... );
    return convolution;
    // engine goes out of scope and gets destroyed
}
