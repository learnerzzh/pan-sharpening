import tensorflow as tf



def D_S(ms,fused,pan,pan_degraded):
    ms = tf.expand_dims(ms,-1)
    fused = tf.expand_dims(fused,-1)

    shape = ms.get_shape().as_list()

    L = shape[3]
    M1 = tf.zeros([L],dtype=tf.float32)
    M2 = tf.zeros([L],dtype=tf.float32)
    for i in range(L):
        A = _uqi_single(fused[:,:,:,i],pan)
        B = _uqi_single(ms[:,:,:,i],pan_degraded)
        one_hot = tf.one_hot(i,8,dtype=tf.float32)
        A_bl = tf.multiply(A,one_hot)
        B_bl = tf.multiply(B,one_hot)
        M1 = tf.add(M1,A_bl)
        M2 = tf.add(M2,B_bl)

    diff = tf.abs(tf.subtract(M1,M2))
    L8 = tf.constant(8,dtype=tf.float32)
    return tf.divide(tf.reduce_sum(diff),L8)

def D_L(ms,fused):
    ms = tf.expand_dims(ms,-1)
    fused = tf.expand_dims(fused,-1)
    shape = fused.get_shape().as_list()
    L = shape[3]
    M1 = tf.zeros((L,L),dtype=fused.dtype)
    M2 = tf.zeros((L,L),dtype=fused.dtype)
    # count = tf.multiply(shape[2],shape[1])
    count = tf.constant(64,dtype=tf.int32)

    for l in range(L):
        for r in range(L):
            A = _uqi_single(fused[:,:,:,l],fused[:,:,:,r])
            B = _uqi_single(ms[:,:,:,l],ms[:,:,:,r])

            M1 = tf.reshape(M1,[1,count])
            one_hot = tf.one_hot(tf.add(tf.multiply(l,shape[3]),r),count,dtype=tf.float32)
            M1 = tf.add(M1,tf.multiply(A,one_hot))
            M1 = tf.reshape(M1,[L,L])

            M2 = tf.reshape(M2,[1,count])
            one_hot = tf.one_hot(tf.add(tf.multiply(l,shape[3]),r),count,dtype=tf.float32)
            M2 = tf.add(M2,tf.multiply(B,one_hot))
            M2 = tf.reshape(M2,[L,L])

    diff = tf.abs(tf.subtract(M1,M2))
    L_f32 = tf.constant(L,dtype=tf.float32)
    e = tf.constant(1,dtype=tf.float32)
    two = tf.constant(2,dtype=tf.float32)
    return tf.multiply(tf.divide(e,tf.multiply(L_f32,tf.subtract(L_f32,e))),tf.multiply(two,tf.reduce_sum(diff)))

def loss_function(y_true,y_pred):
    y_pred = tf.cast(y_pred,tf.float32)
    y_true = tf.cast(y_true,tf.float32)
    ms,pp = tf.split(y_true, [8, 17], 3)
    p1,pp_1 = tf.split(pp, [1, 16], 3)
    p2,pp_2 = tf.split(pp_1, [1, 15], 3)
    p3,pp_3 = tf.split(pp_2, [1, 14], 3)
    p4,pp_4 = tf.split(pp_3, [1, 13], 3)
    p5,pp_5 = tf.split(pp_4, [1, 12], 3)
    p6,pp_6 = tf.split(pp_5, [1, 11], 3)
    p7,pp_7 = tf.split(pp_6, [1, 10], 3)
    p8,pp_8 = tf.split(pp_7, [1, 9], 3)
    p9,pp_9 = tf.split(pp_8, [1, 8], 3)
    p10,pp_10 = tf.split(pp_9, [1, 7], 3)
    p11,pp_11 = tf.split(pp_10, [1, 6], 3)
    p12,pp_12 = tf.split(pp_11, [1, 5], 3)
    p13,pp_13 = tf.split(pp_12, [1, 4], 3)
    p14,pp_14 = tf.split(pp_13, [1, 3], 3)
    p15,pp_15 = tf.split(pp_14, [1, 2], 3)
    p16,p_d = tf.split(pp_15, [1,1], 3)

    s1 = tf.concat([p1, p2],2)
    s2 = tf.concat([s1, p3],2)
    s3 = tf.concat([s2, p4],2)
    z1 = tf.concat([p5, p6],2)
    z2 = tf.concat([z1, p7],2)
    z3 = tf.concat([z2, p8],2)
    x1 = tf.concat([p9, p10],2)
    x2 = tf.concat([x1, p11],2)
    x3 = tf.concat([x2, p12],2)
    d1 = tf.concat([p13, p14],2)
    d2 = tf.concat([d1, p15],2)
    d3 = tf.concat([d2, p16],2)

    pan1 = tf.concat( [s3,z3],1)
    pan2 = tf.concat( [pan1,x3],1)
    pan3 = tf.concat( [pan2,d3],1)

    pan_degraded = p_d
    pan = pan3
    fused = y_pred
    D_s = D_S(ms,fused,pan,pan_degraded)
    D_l=D_L(ms,fused)
    if tf.greater_equal(D_l,D_s) is True :
        result = D_l
    else:
        result = D_s
    # one = tf.constant(1,dtype=tf.float32)
    # qnr = tf.multiply(tf.subtract(one,D_l),tf.subtract(one,D_s))
    # fuyi = tf.constant(-1,dtype=tf.float32)
    # result = tf.multiply(fuyi,qnr)
    return result

def uniform_filter(input,filter_size):
    window = tf.ones([filter_size,filter_size],dtype=tf.float32)
    mask = tf.reshape(window,[filter_size,filter_size,1,1])
    output = tf.nn.conv2d(input,mask,strides=[1,1,1,1], padding='VALID')
    # output = tf.round(output)
    return output

def _uqi_single(GT,P,ws=7):
    shape = GT.get_shape().as_list()
    N = tf.cast(tf.multiply(ws,ws),dtype=tf.float32)
    filter_size = tf.constant(ws,dtype=tf.float32)
    # GT = tf.expand_dims(GT,-1)
    # P = tf.expand_dims(P,-1)

    GT_sq = tf.multiply(GT,GT)
    P_sq = tf.multiply(P,P)
    GT_P = tf.multiply(GT,P)

    GT_sum = uniform_filter(GT, filter_size)
    P_sum =  uniform_filter(P, filter_size)
    GT_sq_sum = uniform_filter(GT_sq, filter_size)
    P_sq_sum = uniform_filter(P_sq, filter_size)
    GT_P_sum = uniform_filter(GT_P, filter_size)

    GT_P_sum_mul = tf.multiply(GT_sum,P_sum)
    GT_P_sum_sq_sum_mul = tf.add(tf.multiply(GT_sum,GT_sum),tf.multiply(P_sum,P_sum))

    four = tf.constant(4,dtype=tf.float32)

    numerator = tf.multiply(four,tf.multiply(tf.subtract(tf.multiply(N,GT_P_sum),GT_P_sum_mul),GT_P_sum_mul))
    denominator1 = tf.subtract(tf.multiply(N,tf.add(GT_sq_sum,P_sq_sum)),GT_P_sum_sq_sum_mul)
    denominator = tf.multiply(denominator1,GT_P_sum_sq_sum_mul)

    denominator = tf.squeeze(denominator)
    numerator =tf.squeeze(numerator)

    index_1 = tf.equal(denominator1,tf.zeros_like(denominator1))
    index_1 = tf.cast(index_1,dtype=tf.float32)
    index_2 = tf.not_equal(GT_P_sum_sq_sum_mul,tf.zeros_like(denominator))
    index_2 = tf.cast(index_2,dtype=tf.float32)
    index = tf.multiply(index_1,index_2)
    two = tf.constant(2,dtype=tf.float32)
    q_map = tf.divide(tf.multiply(two,tf.multiply(GT_P_sum_mul,index)),tf.multiply(GT_P_sum_sq_sum_mul,index))

    index = tf.not_equal(denominator,tf.zeros_like(denominator))
    index = tf.cast(index,dtype=tf.float32)

    numerator_index = tf.multiply(numerator,index)
    denominator_index = tf.multiply(denominator,index)
    q_map = tf.divide(numerator_index,denominator_index)
    juz_one = tf.ones_like(index)
    index_equal_zero = tf.equal(denominator,tf.zeros_like(denominator))
    index_equal_zero = tf.cast(index_equal_zero,dtype=tf.float32)
    denominator_equal_zero_bian_one =tf.multiply(juz_one,index_equal_zero)
    output = tf.add(q_map,denominator_equal_zero_bian_one)
    result = tf.reduce_mean(output)
    return result
