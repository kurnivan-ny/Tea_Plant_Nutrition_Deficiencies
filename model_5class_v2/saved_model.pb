вь
Ќ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
Њ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.3.12v2.3.0-54-gfcc4b966f18ЇЎ
Д
conv2d_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_38/kernel
}
$conv2d_38/kernel/Read/ReadVariableOpReadVariableOpconv2d_38/kernel*&
_output_shapes
:*
dtype0
t
conv2d_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_38/bias
m
"conv2d_38/bias/Read/ReadVariableOpReadVariableOpconv2d_38/bias*
_output_shapes
:*
dtype0
Д
conv2d_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_39/kernel
}
$conv2d_39/kernel/Read/ReadVariableOpReadVariableOpconv2d_39/kernel*&
_output_shapes
:*
dtype0
t
conv2d_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_39/bias
m
"conv2d_39/bias/Read/ReadVariableOpReadVariableOpconv2d_39/bias*
_output_shapes
:*
dtype0
Д
conv2d_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_40/kernel
}
$conv2d_40/kernel/Read/ReadVariableOpReadVariableOpconv2d_40/kernel*&
_output_shapes
: *
dtype0
t
conv2d_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_40/bias
m
"conv2d_40/bias/Read/ReadVariableOpReadVariableOpconv2d_40/bias*
_output_shapes
: *
dtype0
Д
conv2d_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_41/kernel
}
$conv2d_41/kernel/Read/ReadVariableOpReadVariableOpconv2d_41/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_41/bias
m
"conv2d_41/bias/Read/ReadVariableOpReadVariableOpconv2d_41/bias*
_output_shapes
:@*
dtype0
Е
conv2d_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*!
shared_nameconv2d_42/kernel
~
$conv2d_42/kernel/Read/ReadVariableOpReadVariableOpconv2d_42/kernel*'
_output_shapes
:@А*
dtype0
u
conv2d_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_42/bias
n
"conv2d_42/bias/Read/ReadVariableOpReadVariableOpconv2d_42/bias*
_output_shapes	
:А*
dtype0
Ж
conv2d_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*!
shared_nameconv2d_43/kernel

$conv2d_43/kernel/Read/ReadVariableOpReadVariableOpconv2d_43/kernel*(
_output_shapes
:АА*
dtype0
u
conv2d_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_43/bias
n
"conv2d_43/bias/Read/ReadVariableOpReadVariableOpconv2d_43/bias*
_output_shapes	
:А*
dtype0
|
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А и* 
shared_namedense_14/kernel
u
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel* 
_output_shapes
:
А и*
dtype0
s
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:и*
shared_namedense_14/bias
l
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes	
:и*
dtype0
|
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
иф* 
shared_namedense_15/kernel
u
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel* 
_output_shapes
:
иф*
dtype0
s
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ф*
shared_namedense_15/bias
l
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes	
:ф*
dtype0
{
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ф* 
shared_namedense_16/kernel
t
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes
:	ф*
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
Т
Adam/conv2d_38/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_38/kernel/m
Л
+Adam/conv2d_38/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/kernel/m*&
_output_shapes
:*
dtype0
В
Adam/conv2d_38/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_38/bias/m
{
)Adam/conv2d_38/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/bias/m*
_output_shapes
:*
dtype0
Т
Adam/conv2d_39/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_39/kernel/m
Л
+Adam/conv2d_39/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/kernel/m*&
_output_shapes
:*
dtype0
В
Adam/conv2d_39/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_39/bias/m
{
)Adam/conv2d_39/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/bias/m*
_output_shapes
:*
dtype0
Т
Adam/conv2d_40/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_40/kernel/m
Л
+Adam/conv2d_40/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/kernel/m*&
_output_shapes
: *
dtype0
В
Adam/conv2d_40/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_40/bias/m
{
)Adam/conv2d_40/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/bias/m*
_output_shapes
: *
dtype0
Т
Adam/conv2d_41/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_41/kernel/m
Л
+Adam/conv2d_41/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/kernel/m*&
_output_shapes
: @*
dtype0
В
Adam/conv2d_41/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_41/bias/m
{
)Adam/conv2d_41/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/bias/m*
_output_shapes
:@*
dtype0
У
Adam/conv2d_42/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*(
shared_nameAdam/conv2d_42/kernel/m
М
+Adam/conv2d_42/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/kernel/m*'
_output_shapes
:@А*
dtype0
Г
Adam/conv2d_42/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_42/bias/m
|
)Adam/conv2d_42/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/bias/m*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_43/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_43/kernel/m
Н
+Adam/conv2d_43/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/kernel/m*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_43/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_43/bias/m
|
)Adam/conv2d_43/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/bias/m*
_output_shapes	
:А*
dtype0
К
Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А и*'
shared_nameAdam/dense_14/kernel/m
Г
*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m* 
_output_shapes
:
А и*
dtype0
Б
Adam/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:и*%
shared_nameAdam/dense_14/bias/m
z
(Adam/dense_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/m*
_output_shapes	
:и*
dtype0
К
Adam/dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
иф*'
shared_nameAdam/dense_15/kernel/m
Г
*Adam/dense_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/m* 
_output_shapes
:
иф*
dtype0
Б
Adam/dense_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ф*%
shared_nameAdam/dense_15/bias/m
z
(Adam/dense_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/m*
_output_shapes	
:ф*
dtype0
Й
Adam/dense_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ф*'
shared_nameAdam/dense_16/kernel/m
В
*Adam/dense_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/m*
_output_shapes
:	ф*
dtype0
А
Adam/dense_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_16/bias/m
y
(Adam/dense_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/m*
_output_shapes
:*
dtype0
Т
Adam/conv2d_38/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_38/kernel/v
Л
+Adam/conv2d_38/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/kernel/v*&
_output_shapes
:*
dtype0
В
Adam/conv2d_38/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_38/bias/v
{
)Adam/conv2d_38/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/bias/v*
_output_shapes
:*
dtype0
Т
Adam/conv2d_39/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_39/kernel/v
Л
+Adam/conv2d_39/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/kernel/v*&
_output_shapes
:*
dtype0
В
Adam/conv2d_39/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_39/bias/v
{
)Adam/conv2d_39/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/bias/v*
_output_shapes
:*
dtype0
Т
Adam/conv2d_40/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_40/kernel/v
Л
+Adam/conv2d_40/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/kernel/v*&
_output_shapes
: *
dtype0
В
Adam/conv2d_40/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_40/bias/v
{
)Adam/conv2d_40/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/bias/v*
_output_shapes
: *
dtype0
Т
Adam/conv2d_41/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_41/kernel/v
Л
+Adam/conv2d_41/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/kernel/v*&
_output_shapes
: @*
dtype0
В
Adam/conv2d_41/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_41/bias/v
{
)Adam/conv2d_41/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/bias/v*
_output_shapes
:@*
dtype0
У
Adam/conv2d_42/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*(
shared_nameAdam/conv2d_42/kernel/v
М
+Adam/conv2d_42/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/kernel/v*'
_output_shapes
:@А*
dtype0
Г
Adam/conv2d_42/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_42/bias/v
|
)Adam/conv2d_42/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/bias/v*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_43/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_43/kernel/v
Н
+Adam/conv2d_43/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/kernel/v*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_43/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_43/bias/v
|
)Adam/conv2d_43/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/bias/v*
_output_shapes	
:А*
dtype0
К
Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А и*'
shared_nameAdam/dense_14/kernel/v
Г
*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v* 
_output_shapes
:
А и*
dtype0
Б
Adam/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:и*%
shared_nameAdam/dense_14/bias/v
z
(Adam/dense_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/v*
_output_shapes	
:и*
dtype0
К
Adam/dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
иф*'
shared_nameAdam/dense_15/kernel/v
Г
*Adam/dense_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/v* 
_output_shapes
:
иф*
dtype0
Б
Adam/dense_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ф*%
shared_nameAdam/dense_15/bias/v
z
(Adam/dense_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/v*
_output_shapes	
:ф*
dtype0
Й
Adam/dense_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ф*'
shared_nameAdam/dense_16/kernel/v
В
*Adam/dense_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/v*
_output_shapes
:	ф*
dtype0
А
Adam/dense_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_16/bias/v
y
(Adam/dense_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
рГ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*™Г
valueЯГBЫГ BУГ
љ
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer_with_weights-4
layer-12
layer-13
layer-14
layer_with_weights-5
layer-15
layer-16
layer-17
layer-18
layer_with_weights-6
layer-19
layer-20
layer_with_weights-7
layer-21
layer-22
layer-23
layer_with_weights-8
layer-24
layer-25
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api
 
signatures
h

!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api
R
'trainable_variables
(	variables
)regularization_losses
*	keras_api
R
+trainable_variables
,	variables
-regularization_losses
.	keras_api
h

/kernel
0bias
1trainable_variables
2	variables
3regularization_losses
4	keras_api
R
5trainable_variables
6	variables
7regularization_losses
8	keras_api
R
9trainable_variables
:	variables
;regularization_losses
<	keras_api
h

=kernel
>bias
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
R
Ctrainable_variables
D	variables
Eregularization_losses
F	keras_api
R
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
h

Kkernel
Lbias
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
R
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
R
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
h

Ykernel
Zbias
[trainable_variables
\	variables
]regularization_losses
^	keras_api
R
_trainable_variables
`	variables
aregularization_losses
b	keras_api
R
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
h

gkernel
hbias
itrainable_variables
j	variables
kregularization_losses
l	keras_api
R
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
R
qtrainable_variables
r	variables
sregularization_losses
t	keras_api
R
utrainable_variables
v	variables
wregularization_losses
x	keras_api
h

ykernel
zbias
{trainable_variables
|	variables
}regularization_losses
~	keras_api
U
trainable_variables
А	variables
Бregularization_losses
В	keras_api
n
Гkernel
	Дbias
Еtrainable_variables
Ж	variables
Зregularization_losses
И	keras_api
V
Йtrainable_variables
К	variables
Лregularization_losses
М	keras_api
V
Нtrainable_variables
О	variables
Пregularization_losses
Р	keras_api
n
Сkernel
	Тbias
Уtrainable_variables
Ф	variables
Хregularization_losses
Ц	keras_api
V
Чtrainable_variables
Ш	variables
Щregularization_losses
Ъ	keras_api
µ
	Ыiter
Ьbeta_1
Эbeta_2

Юdecay
Яlearning_rate!m≤"m≥/mі0mµ=mґ>mЈKmЄLmєYmЇZmїgmЉhmљymЊzmњ	Гmј	ДmЅ	Сm¬	Тm√!vƒ"v≈/v∆0v«=v»>v…Kv LvЋYvћZvЌgvќhvѕyv–zv—	Гv“	Дv”	Сv‘	Тv’
К
!0
"1
/2
03
=4
>5
K6
L7
Y8
Z9
g10
h11
y12
z13
Г14
Д15
С16
Т17
К
!0
"1
/2
03
=4
>5
K6
L7
Y8
Z9
g10
h11
y12
z13
Г14
Д15
С16
Т17
 
≤
trainable_variables
	variables
†non_trainable_variables
°metrics
Ґlayers
 £layer_regularization_losses
regularization_losses
§layer_metrics
 
\Z
VARIABLE_VALUEconv2d_38/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_38/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
≤
#trainable_variables
$	variables
•metrics
¶non_trainable_variables
Іlayers
 ®layer_regularization_losses
%regularization_losses
©layer_metrics
 
 
 
≤
'trainable_variables
(	variables
™metrics
Ђnon_trainable_variables
ђlayers
 ≠layer_regularization_losses
)regularization_losses
Ѓlayer_metrics
 
 
 
≤
+trainable_variables
,	variables
ѓmetrics
∞non_trainable_variables
±layers
 ≤layer_regularization_losses
-regularization_losses
≥layer_metrics
\Z
VARIABLE_VALUEconv2d_39/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_39/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

/0
01

/0
01
 
≤
1trainable_variables
2	variables
іmetrics
µnon_trainable_variables
ґlayers
 Јlayer_regularization_losses
3regularization_losses
Єlayer_metrics
 
 
 
≤
5trainable_variables
6	variables
єmetrics
Їnon_trainable_variables
їlayers
 Љlayer_regularization_losses
7regularization_losses
љlayer_metrics
 
 
 
≤
9trainable_variables
:	variables
Њmetrics
њnon_trainable_variables
јlayers
 Ѕlayer_regularization_losses
;regularization_losses
¬layer_metrics
\Z
VARIABLE_VALUEconv2d_40/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_40/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

=0
>1

=0
>1
 
≤
?trainable_variables
@	variables
√metrics
ƒnon_trainable_variables
≈layers
 ∆layer_regularization_losses
Aregularization_losses
«layer_metrics
 
 
 
≤
Ctrainable_variables
D	variables
»metrics
…non_trainable_variables
 layers
 Ћlayer_regularization_losses
Eregularization_losses
ћlayer_metrics
 
 
 
≤
Gtrainable_variables
H	variables
Ќmetrics
ќnon_trainable_variables
ѕlayers
 –layer_regularization_losses
Iregularization_losses
—layer_metrics
\Z
VARIABLE_VALUEconv2d_41/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_41/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

K0
L1

K0
L1
 
≤
Mtrainable_variables
N	variables
“metrics
”non_trainable_variables
‘layers
 ’layer_regularization_losses
Oregularization_losses
÷layer_metrics
 
 
 
≤
Qtrainable_variables
R	variables
„metrics
Ўnon_trainable_variables
ўlayers
 Џlayer_regularization_losses
Sregularization_losses
џlayer_metrics
 
 
 
≤
Utrainable_variables
V	variables
№metrics
Ёnon_trainable_variables
ёlayers
 яlayer_regularization_losses
Wregularization_losses
аlayer_metrics
\Z
VARIABLE_VALUEconv2d_42/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_42/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1

Y0
Z1
 
≤
[trainable_variables
\	variables
бmetrics
вnon_trainable_variables
гlayers
 дlayer_regularization_losses
]regularization_losses
еlayer_metrics
 
 
 
≤
_trainable_variables
`	variables
жmetrics
зnon_trainable_variables
иlayers
 йlayer_regularization_losses
aregularization_losses
кlayer_metrics
 
 
 
≤
ctrainable_variables
d	variables
лmetrics
мnon_trainable_variables
нlayers
 оlayer_regularization_losses
eregularization_losses
пlayer_metrics
\Z
VARIABLE_VALUEconv2d_43/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_43/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

g0
h1

g0
h1
 
≤
itrainable_variables
j	variables
рmetrics
сnon_trainable_variables
тlayers
 уlayer_regularization_losses
kregularization_losses
фlayer_metrics
 
 
 
≤
mtrainable_variables
n	variables
хmetrics
цnon_trainable_variables
чlayers
 шlayer_regularization_losses
oregularization_losses
щlayer_metrics
 
 
 
≤
qtrainable_variables
r	variables
ъmetrics
ыnon_trainable_variables
ьlayers
 эlayer_regularization_losses
sregularization_losses
юlayer_metrics
 
 
 
≤
utrainable_variables
v	variables
€metrics
Аnon_trainable_variables
Бlayers
 Вlayer_regularization_losses
wregularization_losses
Гlayer_metrics
[Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_14/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

y0
z1

y0
z1
 
≤
{trainable_variables
|	variables
Дmetrics
Еnon_trainable_variables
Жlayers
 Зlayer_regularization_losses
}regularization_losses
Иlayer_metrics
 
 
 
і
trainable_variables
А	variables
Йmetrics
Кnon_trainable_variables
Лlayers
 Мlayer_regularization_losses
Бregularization_losses
Нlayer_metrics
[Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_15/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

Г0
Д1

Г0
Д1
 
µ
Еtrainable_variables
Ж	variables
Оmetrics
Пnon_trainable_variables
Рlayers
 Сlayer_regularization_losses
Зregularization_losses
Тlayer_metrics
 
 
 
µ
Йtrainable_variables
К	variables
Уmetrics
Фnon_trainable_variables
Хlayers
 Цlayer_regularization_losses
Лregularization_losses
Чlayer_metrics
 
 
 
µ
Нtrainable_variables
О	variables
Шmetrics
Щnon_trainable_variables
Ъlayers
 Ыlayer_regularization_losses
Пregularization_losses
Ьlayer_metrics
[Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_16/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

С0
Т1

С0
Т1
 
µ
Уtrainable_variables
Ф	variables
Эmetrics
Юnon_trainable_variables
Яlayers
 †layer_regularization_losses
Хregularization_losses
°layer_metrics
 
 
 
µ
Чtrainable_variables
Ш	variables
Ґmetrics
£non_trainable_variables
§layers
 •layer_regularization_losses
Щregularization_losses
¶layer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

І0
®1
∆
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

©total

™count
Ђ	variables
ђ	keras_api
I

≠total

Ѓcount
ѓ
_fn_kwargs
∞	variables
±	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

©0
™1

Ђ	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

≠0
Ѓ1

∞	variables
}
VARIABLE_VALUEAdam/conv2d_38/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_38/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_39/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_39/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_40/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_40/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_41/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_41/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_42/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_42/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_43/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_43/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_14/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_14/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_15/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_15/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_16/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_16/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_38/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_38/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_39/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_39/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_40/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_40/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_41/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_41/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_42/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_42/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_43/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_43/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_14/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_14/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_15/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_15/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_16/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_16/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ц
serving_default_conv2d_38_inputPlaceholder*1
_output_shapes
:€€€€€€€€€АА*
dtype0*&
shape:€€€€€€€€€АА
Й
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_38_inputconv2d_38/kernelconv2d_38/biasconv2d_39/kernelconv2d_39/biasconv2d_40/kernelconv2d_40/biasconv2d_41/kernelconv2d_41/biasconv2d_42/kernelconv2d_42/biasconv2d_43/kernelconv2d_43/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_29193
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
™
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_38/kernel/Read/ReadVariableOp"conv2d_38/bias/Read/ReadVariableOp$conv2d_39/kernel/Read/ReadVariableOp"conv2d_39/bias/Read/ReadVariableOp$conv2d_40/kernel/Read/ReadVariableOp"conv2d_40/bias/Read/ReadVariableOp$conv2d_41/kernel/Read/ReadVariableOp"conv2d_41/bias/Read/ReadVariableOp$conv2d_42/kernel/Read/ReadVariableOp"conv2d_42/bias/Read/ReadVariableOp$conv2d_43/kernel/Read/ReadVariableOp"conv2d_43/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_38/kernel/m/Read/ReadVariableOp)Adam/conv2d_38/bias/m/Read/ReadVariableOp+Adam/conv2d_39/kernel/m/Read/ReadVariableOp)Adam/conv2d_39/bias/m/Read/ReadVariableOp+Adam/conv2d_40/kernel/m/Read/ReadVariableOp)Adam/conv2d_40/bias/m/Read/ReadVariableOp+Adam/conv2d_41/kernel/m/Read/ReadVariableOp)Adam/conv2d_41/bias/m/Read/ReadVariableOp+Adam/conv2d_42/kernel/m/Read/ReadVariableOp)Adam/conv2d_42/bias/m/Read/ReadVariableOp+Adam/conv2d_43/kernel/m/Read/ReadVariableOp)Adam/conv2d_43/bias/m/Read/ReadVariableOp*Adam/dense_14/kernel/m/Read/ReadVariableOp(Adam/dense_14/bias/m/Read/ReadVariableOp*Adam/dense_15/kernel/m/Read/ReadVariableOp(Adam/dense_15/bias/m/Read/ReadVariableOp*Adam/dense_16/kernel/m/Read/ReadVariableOp(Adam/dense_16/bias/m/Read/ReadVariableOp+Adam/conv2d_38/kernel/v/Read/ReadVariableOp)Adam/conv2d_38/bias/v/Read/ReadVariableOp+Adam/conv2d_39/kernel/v/Read/ReadVariableOp)Adam/conv2d_39/bias/v/Read/ReadVariableOp+Adam/conv2d_40/kernel/v/Read/ReadVariableOp)Adam/conv2d_40/bias/v/Read/ReadVariableOp+Adam/conv2d_41/kernel/v/Read/ReadVariableOp)Adam/conv2d_41/bias/v/Read/ReadVariableOp+Adam/conv2d_42/kernel/v/Read/ReadVariableOp)Adam/conv2d_42/bias/v/Read/ReadVariableOp+Adam/conv2d_43/kernel/v/Read/ReadVariableOp)Adam/conv2d_43/bias/v/Read/ReadVariableOp*Adam/dense_14/kernel/v/Read/ReadVariableOp(Adam/dense_14/bias/v/Read/ReadVariableOp*Adam/dense_15/kernel/v/Read/ReadVariableOp(Adam/dense_15/bias/v/Read/ReadVariableOp*Adam/dense_16/kernel/v/Read/ReadVariableOp(Adam/dense_16/bias/v/Read/ReadVariableOpConst*L
TinE
C2A	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__traced_save_29945
є
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_38/kernelconv2d_38/biasconv2d_39/kernelconv2d_39/biasconv2d_40/kernelconv2d_40/biasconv2d_41/kernelconv2d_41/biasconv2d_42/kernelconv2d_42/biasconv2d_43/kernelconv2d_43/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_38/kernel/mAdam/conv2d_38/bias/mAdam/conv2d_39/kernel/mAdam/conv2d_39/bias/mAdam/conv2d_40/kernel/mAdam/conv2d_40/bias/mAdam/conv2d_41/kernel/mAdam/conv2d_41/bias/mAdam/conv2d_42/kernel/mAdam/conv2d_42/bias/mAdam/conv2d_43/kernel/mAdam/conv2d_43/bias/mAdam/dense_14/kernel/mAdam/dense_14/bias/mAdam/dense_15/kernel/mAdam/dense_15/bias/mAdam/dense_16/kernel/mAdam/dense_16/bias/mAdam/conv2d_38/kernel/vAdam/conv2d_38/bias/vAdam/conv2d_39/kernel/vAdam/conv2d_39/bias/vAdam/conv2d_40/kernel/vAdam/conv2d_40/bias/vAdam/conv2d_41/kernel/vAdam/conv2d_41/bias/vAdam/conv2d_42/kernel/vAdam/conv2d_42/bias/vAdam/conv2d_43/kernel/vAdam/conv2d_43/bias/vAdam/dense_14/kernel/vAdam/dense_14/bias/vAdam/dense_15/kernel/vAdam/dense_15/bias/vAdam/dense_16/kernel/vAdam/dense_16/bias/v*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_restore_30144л≠
™
ђ
D__inference_conv2d_42_layer_call_and_return_conditional_losses_29560

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@:::W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ъ
I
-__inference_activation_58_layer_call_fn_29733

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_58_layer_call_and_return_conditional_losses_288522
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
≠
ђ
D__inference_conv2d_43_layer_call_and_return_conditional_losses_29589

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€А:::X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
÷
d
H__inference_activation_52_layer_call_and_return_conditional_losses_28570

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@@ :W S
/
_output_shapes
:€€€€€€€€€@@ 
 
_user_specified_nameinputs
Шf
њ
G__inference_sequential_7_layer_call_and_return_conditional_losses_28996

inputs
conv2d_38_28933
conv2d_38_28935
conv2d_39_28940
conv2d_39_28942
conv2d_40_28947
conv2d_40_28949
conv2d_41_28954
conv2d_41_28956
conv2d_42_28961
conv2d_42_28963
conv2d_43_28968
conv2d_43_28970
dense_14_28976
dense_14_28978
dense_15_28982
dense_15_28984
dense_16_28989
dense_16_28991
identityИҐ!conv2d_38/StatefulPartitionedCallҐ!conv2d_39/StatefulPartitionedCallҐ!conv2d_40/StatefulPartitionedCallҐ!conv2d_41/StatefulPartitionedCallҐ!conv2d_42/StatefulPartitionedCallҐ!conv2d_43/StatefulPartitionedCallҐ dense_14/StatefulPartitionedCallҐ dense_15/StatefulPartitionedCallҐ dense_16/StatefulPartitionedCallҐ!dropout_4/StatefulPartitionedCall†
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_38_28933conv2d_38_28935*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_38_layer_call_and_return_conditional_losses_284692#
!conv2d_38/StatefulPartitionedCallР
activation_50/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_50_layer_call_and_return_conditional_losses_284902
activation_50/PartitionedCallХ
 max_pooling2d_38/PartitionedCallPartitionedCall&activation_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_283892"
 max_pooling2d_38/PartitionedCall√
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_38/PartitionedCall:output:0conv2d_39_28940conv2d_39_28942*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_39_layer_call_and_return_conditional_losses_285092#
!conv2d_39/StatefulPartitionedCallР
activation_51/PartitionedCallPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_51_layer_call_and_return_conditional_losses_285302
activation_51/PartitionedCallУ
 max_pooling2d_39/PartitionedCallPartitionedCall&activation_51/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_284012"
 max_pooling2d_39/PartitionedCallЅ
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_39/PartitionedCall:output:0conv2d_40_28947conv2d_40_28949*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_40_layer_call_and_return_conditional_losses_285492#
!conv2d_40/StatefulPartitionedCallО
activation_52/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_52_layer_call_and_return_conditional_losses_285702
activation_52/PartitionedCallУ
 max_pooling2d_40/PartitionedCallPartitionedCall&activation_52/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_284132"
 max_pooling2d_40/PartitionedCallЅ
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_40/PartitionedCall:output:0conv2d_41_28954conv2d_41_28956*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_285892#
!conv2d_41/StatefulPartitionedCallО
activation_53/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_53_layer_call_and_return_conditional_losses_286102
activation_53/PartitionedCallУ
 max_pooling2d_41/PartitionedCallPartitionedCall&activation_53/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_284252"
 max_pooling2d_41/PartitionedCall¬
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_41/PartitionedCall:output:0conv2d_42_28961conv2d_42_28963*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_42_layer_call_and_return_conditional_losses_286292#
!conv2d_42/StatefulPartitionedCallП
activation_54/PartitionedCallPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_54_layer_call_and_return_conditional_losses_286502
activation_54/PartitionedCallФ
 max_pooling2d_42/PartitionedCallPartitionedCall&activation_54/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_284372"
 max_pooling2d_42/PartitionedCall¬
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_42/PartitionedCall:output:0conv2d_43_28968conv2d_43_28970*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_43_layer_call_and_return_conditional_losses_286692#
!conv2d_43/StatefulPartitionedCallП
activation_55/PartitionedCallPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_55_layer_call_and_return_conditional_losses_286902
activation_55/PartitionedCallФ
 max_pooling2d_43/PartitionedCallPartitionedCall&activation_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_43_layer_call_and_return_conditional_losses_284492"
 max_pooling2d_43/PartitionedCallъ
flatten_6/PartitionedCallPartitionedCall)max_pooling2d_43/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_287052
flatten_6/PartitionedCallЃ
 dense_14/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_14_28976dense_14_28978*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_287232"
 dense_14/StatefulPartitionedCallЖ
activation_56/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_56_layer_call_and_return_conditional_losses_287442
activation_56/PartitionedCall≤
 dense_15/StatefulPartitionedCallStatefulPartitionedCall&activation_56/PartitionedCall:output:0dense_15_28982dense_15_28984*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_287622"
 dense_15/StatefulPartitionedCallТ
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_287902#
!dropout_4/StatefulPartitionedCallЗ
activation_57/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_57_layer_call_and_return_conditional_losses_288132
activation_57/PartitionedCall±
 dense_16/StatefulPartitionedCallStatefulPartitionedCall&activation_57/PartitionedCall:output:0dense_16_28989dense_16_28991*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_288312"
 dense_16/StatefulPartitionedCallЕ
activation_58/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_58_layer_call_and_return_conditional_losses_288522
activation_58/PartitionedCallя
IdentityIdentity&activation_58/PartitionedCall:output:0"^conv2d_38/StatefulPartitionedCall"^conv2d_39/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА::::::::::::::::::2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
ѕ
Ђ
C__inference_dense_16_layer_call_and_return_conditional_losses_28831

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ф*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€ф:::P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
Џ
d
H__inference_activation_55_layer_call_and_return_conditional_losses_29603

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:€€€€€€€€€А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Њ]
•
G__inference_sequential_7_layer_call_and_return_conditional_losses_29276

inputs,
(conv2d_38_conv2d_readvariableop_resource-
)conv2d_38_biasadd_readvariableop_resource,
(conv2d_39_conv2d_readvariableop_resource-
)conv2d_39_biasadd_readvariableop_resource,
(conv2d_40_conv2d_readvariableop_resource-
)conv2d_40_biasadd_readvariableop_resource,
(conv2d_41_conv2d_readvariableop_resource-
)conv2d_41_biasadd_readvariableop_resource,
(conv2d_42_conv2d_readvariableop_resource-
)conv2d_42_biasadd_readvariableop_resource,
(conv2d_43_conv2d_readvariableop_resource-
)conv2d_43_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource
identityИ≥
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_38/Conv2D/ReadVariableOp√
conv2d_38/Conv2DConv2Dinputs'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingSAME*
strides
2
conv2d_38/Conv2D™
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp≤
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
conv2d_38/BiasAddИ
activation_50/ReluReluconv2d_38/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
activation_50/Relu–
max_pooling2d_38/MaxPoolMaxPool activation_50/Relu:activations:0*1
_output_shapes
:€€€€€€€€€АА*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPool≥
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_39/Conv2D/ReadVariableOpё
conv2d_39/Conv2DConv2D!max_pooling2d_38/MaxPool:output:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingSAME*
strides
2
conv2d_39/Conv2D™
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp≤
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
conv2d_39/BiasAddИ
activation_51/ReluReluconv2d_39/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
activation_51/Reluќ
max_pooling2d_39/MaxPoolMaxPool activation_51/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPool≥
conv2d_40/Conv2D/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_40/Conv2D/ReadVariableOp№
conv2d_40/Conv2DConv2D!max_pooling2d_39/MaxPool:output:0'conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
paddingSAME*
strides
2
conv2d_40/Conv2D™
 conv2d_40/BiasAdd/ReadVariableOpReadVariableOp)conv2d_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_40/BiasAdd/ReadVariableOp∞
conv2d_40/BiasAddBiasAddconv2d_40/Conv2D:output:0(conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
conv2d_40/BiasAddЖ
activation_52/ReluReluconv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
activation_52/Reluќ
max_pooling2d_40/MaxPoolMaxPool activation_52/Relu:activations:0*/
_output_shapes
:€€€€€€€€€   *
ksize
*
paddingVALID*
strides
2
max_pooling2d_40/MaxPool≥
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_41/Conv2D/ReadVariableOp№
conv2d_41/Conv2DConv2D!max_pooling2d_40/MaxPool:output:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingSAME*
strides
2
conv2d_41/Conv2D™
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_41/BiasAdd/ReadVariableOp∞
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @2
conv2d_41/BiasAddЖ
activation_53/ReluReluconv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€  @2
activation_53/Reluќ
max_pooling2d_41/MaxPoolMaxPool activation_53/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_41/MaxPoolі
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02!
conv2d_42/Conv2D/ReadVariableOpЁ
conv2d_42/Conv2DConv2D!max_pooling2d_41/MaxPool:output:0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_42/Conv2DЂ
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_42/BiasAdd/ReadVariableOp±
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_42/BiasAddЗ
activation_54/ReluReluconv2d_42/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
activation_54/Reluѕ
max_pooling2d_42/MaxPoolMaxPool activation_54/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_42/MaxPoolµ
conv2d_43/Conv2D/ReadVariableOpReadVariableOp(conv2d_43_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_43/Conv2D/ReadVariableOpЁ
conv2d_43/Conv2DConv2D!max_pooling2d_42/MaxPool:output:0'conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_43/Conv2DЂ
 conv2d_43/BiasAdd/ReadVariableOpReadVariableOp)conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_43/BiasAdd/ReadVariableOp±
conv2d_43/BiasAddBiasAddconv2d_43/Conv2D:output:0(conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_43/BiasAddЗ
activation_55/ReluReluconv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
activation_55/Reluѕ
max_pooling2d_43/MaxPoolMaxPool activation_55/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_43/MaxPools
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten_6/Const°
flatten_6/ReshapeReshape!max_pooling2d_43/MaxPool:output:0flatten_6/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А 2
flatten_6/Reshape™
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А и*
dtype02 
dense_14/MatMul/ReadVariableOp£
dense_14/MatMulMatMulflatten_6/Reshape:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€и2
dense_14/MatMul®
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:и*
dtype02!
dense_14/BiasAdd/ReadVariableOp¶
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€и2
dense_14/BiasAdd~
activation_56/ReluReludense_14/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€и2
activation_56/Relu™
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource* 
_output_shapes
:
иф*
dtype02 
dense_15/MatMul/ReadVariableOp©
dense_15/MatMulMatMul activation_56/Relu:activations:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dense_15/MatMul®
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:ф*
dtype02!
dense_15/BiasAdd/ReadVariableOp¶
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dense_15/BiasAddw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout_4/dropout/Const•
dropout_4/dropout/MulMuldense_15/BiasAdd:output:0 dropout_4/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dropout_4/dropout/Mul{
dropout_4/dropout/ShapeShapedense_15/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape”
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф*
dtype020
.dropout_4/dropout/random_uniform/RandomUniformЙ
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2"
 dropout_4/dropout/GreaterEqual/yз
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2 
dropout_4/dropout/GreaterEqualЮ
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ф2
dropout_4/dropout/Cast£
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dropout_4/dropout/Mul_1А
activation_57/ReluReludropout_4/dropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
activation_57/Relu©
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes
:	ф*
dtype02 
dense_16/MatMul/ReadVariableOp®
dense_16/MatMulMatMul activation_57/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_16/MatMulІ
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_16/BiasAdd/ReadVariableOp•
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_16/BiasAddЖ
activation_58/SoftmaxSoftmaxdense_16/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
activation_58/Softmaxs
IdentityIdentityactivation_58/Softmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА:::::::::::::::::::Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
ё
d
H__inference_activation_51_layer_call_and_return_conditional_losses_28530

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:€€€€€€€€€АА2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Њ
d
H__inference_activation_58_layer_call_and_return_conditional_losses_28852

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:€€€€€€€€€2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
•
ђ
D__inference_conv2d_41_layer_call_and_return_conditional_losses_28589

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€   :::W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs
ё
d
H__inference_activation_50_layer_call_and_return_conditional_losses_29458

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:€€€€€€€€€АА2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
•
ђ
D__inference_conv2d_40_layer_call_and_return_conditional_losses_29502

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@:::W S
/
_output_shapes
:€€€€€€€€€@@
 
_user_specified_nameinputs
≠
L
0__inference_max_pooling2d_43_layer_call_fn_28455

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_43_layer_call_and_return_conditional_losses_284492
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Њ
I
-__inference_activation_54_layer_call_fn_29579

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_54_layer_call_and_return_conditional_losses_286502
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
≥f
»
G__inference_sequential_7_layer_call_and_return_conditional_losses_28861
conv2d_38_input
conv2d_38_28480
conv2d_38_28482
conv2d_39_28520
conv2d_39_28522
conv2d_40_28560
conv2d_40_28562
conv2d_41_28600
conv2d_41_28602
conv2d_42_28640
conv2d_42_28642
conv2d_43_28680
conv2d_43_28682
dense_14_28734
dense_14_28736
dense_15_28773
dense_15_28775
dense_16_28842
dense_16_28844
identityИҐ!conv2d_38/StatefulPartitionedCallҐ!conv2d_39/StatefulPartitionedCallҐ!conv2d_40/StatefulPartitionedCallҐ!conv2d_41/StatefulPartitionedCallҐ!conv2d_42/StatefulPartitionedCallҐ!conv2d_43/StatefulPartitionedCallҐ dense_14/StatefulPartitionedCallҐ dense_15/StatefulPartitionedCallҐ dense_16/StatefulPartitionedCallҐ!dropout_4/StatefulPartitionedCall©
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCallconv2d_38_inputconv2d_38_28480conv2d_38_28482*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_38_layer_call_and_return_conditional_losses_284692#
!conv2d_38/StatefulPartitionedCallР
activation_50/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_50_layer_call_and_return_conditional_losses_284902
activation_50/PartitionedCallХ
 max_pooling2d_38/PartitionedCallPartitionedCall&activation_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_283892"
 max_pooling2d_38/PartitionedCall√
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_38/PartitionedCall:output:0conv2d_39_28520conv2d_39_28522*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_39_layer_call_and_return_conditional_losses_285092#
!conv2d_39/StatefulPartitionedCallР
activation_51/PartitionedCallPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_51_layer_call_and_return_conditional_losses_285302
activation_51/PartitionedCallУ
 max_pooling2d_39/PartitionedCallPartitionedCall&activation_51/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_284012"
 max_pooling2d_39/PartitionedCallЅ
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_39/PartitionedCall:output:0conv2d_40_28560conv2d_40_28562*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_40_layer_call_and_return_conditional_losses_285492#
!conv2d_40/StatefulPartitionedCallО
activation_52/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_52_layer_call_and_return_conditional_losses_285702
activation_52/PartitionedCallУ
 max_pooling2d_40/PartitionedCallPartitionedCall&activation_52/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_284132"
 max_pooling2d_40/PartitionedCallЅ
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_40/PartitionedCall:output:0conv2d_41_28600conv2d_41_28602*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_285892#
!conv2d_41/StatefulPartitionedCallО
activation_53/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_53_layer_call_and_return_conditional_losses_286102
activation_53/PartitionedCallУ
 max_pooling2d_41/PartitionedCallPartitionedCall&activation_53/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_284252"
 max_pooling2d_41/PartitionedCall¬
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_41/PartitionedCall:output:0conv2d_42_28640conv2d_42_28642*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_42_layer_call_and_return_conditional_losses_286292#
!conv2d_42/StatefulPartitionedCallП
activation_54/PartitionedCallPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_54_layer_call_and_return_conditional_losses_286502
activation_54/PartitionedCallФ
 max_pooling2d_42/PartitionedCallPartitionedCall&activation_54/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_284372"
 max_pooling2d_42/PartitionedCall¬
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_42/PartitionedCall:output:0conv2d_43_28680conv2d_43_28682*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_43_layer_call_and_return_conditional_losses_286692#
!conv2d_43/StatefulPartitionedCallП
activation_55/PartitionedCallPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_55_layer_call_and_return_conditional_losses_286902
activation_55/PartitionedCallФ
 max_pooling2d_43/PartitionedCallPartitionedCall&activation_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_43_layer_call_and_return_conditional_losses_284492"
 max_pooling2d_43/PartitionedCallъ
flatten_6/PartitionedCallPartitionedCall)max_pooling2d_43/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_287052
flatten_6/PartitionedCallЃ
 dense_14/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_14_28734dense_14_28736*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_287232"
 dense_14/StatefulPartitionedCallЖ
activation_56/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_56_layer_call_and_return_conditional_losses_287442
activation_56/PartitionedCall≤
 dense_15/StatefulPartitionedCallStatefulPartitionedCall&activation_56/PartitionedCall:output:0dense_15_28773dense_15_28775*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_287622"
 dense_15/StatefulPartitionedCallТ
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_287902#
!dropout_4/StatefulPartitionedCallЗ
activation_57/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_57_layer_call_and_return_conditional_losses_288132
activation_57/PartitionedCall±
 dense_16/StatefulPartitionedCallStatefulPartitionedCall&activation_57/PartitionedCall:output:0dense_16_28842dense_16_28844*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_288312"
 dense_16/StatefulPartitionedCallЕ
activation_58/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_58_layer_call_and_return_conditional_losses_288522
activation_58/PartitionedCallя
IdentityIdentity&activation_58/PartitionedCall:output:0"^conv2d_38/StatefulPartitionedCall"^conv2d_39/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА::::::::::::::::::2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall:b ^
1
_output_shapes
:€€€€€€€€€АА
)
_user_specified_nameconv2d_38_input
№
}
(__inference_dense_16_layer_call_fn_29723

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_288312
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€ф::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
Џ
d
H__inference_activation_54_layer_call_and_return_conditional_losses_29574

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:€€€€€€€€€А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Е
ш
,__inference_sequential_7_layer_call_fn_29434

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identityИҐStatefulPartitionedCallќ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_7_layer_call_and_return_conditional_losses_291032
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
≠
ђ
D__inference_conv2d_43_layer_call_and_return_conditional_losses_28669

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€А:::X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
÷
d
H__inference_activation_52_layer_call_and_return_conditional_losses_29516

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@@ :W S
/
_output_shapes
:€€€€€€€€€@@ 
 
_user_specified_nameinputs
ѓ
ђ
D__inference_conv2d_39_layer_call_and_return_conditional_losses_28509

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp•
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА2	
BiasAddn
IdentityIdentityBiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА:::Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
÷
d
H__inference_activation_53_layer_call_and_return_conditional_losses_29545

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€  @2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€  @2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€  @:W S
/
_output_shapes
:€€€€€€€€€  @
 
_user_specified_nameinputs
А
~
)__inference_conv2d_43_layer_call_fn_29598

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_43_layer_call_and_return_conditional_losses_286692
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
‘
Ђ
C__inference_dense_14_layer_call_and_return_conditional_losses_29629

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А и*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€и2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:и*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€и2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€и2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А :::P L
(
_output_shapes
:€€€€€€€€€А 
 
_user_specified_nameinputs
Б
g
K__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_28437

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Њ
I
-__inference_activation_55_layer_call_fn_29608

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_55_layer_call_and_return_conditional_losses_286902
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
≠
L
0__inference_max_pooling2d_38_layer_call_fn_28395

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_283892
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ц
E
)__inference_dropout_4_layer_call_fn_29694

inputs
identity√
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_287952
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€ф:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
ь
~
)__inference_conv2d_40_layer_call_fn_29511

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_40_layer_call_and_return_conditional_losses_285492
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@
 
_user_specified_nameinputs
Ћ
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_28795

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:€€€€€€€€€ф:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
Ї
d
H__inference_activation_57_layer_call_and_return_conditional_losses_28813

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:€€€€€€€€€ф2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€ф:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
ѕ
Ђ
C__inference_dense_16_layer_call_and_return_conditional_losses_29714

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ф*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€ф:::P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
Њ
d
H__inference_activation_58_layer_call_and_return_conditional_losses_29728

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:€€€€€€€€€2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ё
d
H__inference_activation_50_layer_call_and_return_conditional_losses_28490

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:€€€€€€€€€АА2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
•
ђ
D__inference_conv2d_41_layer_call_and_return_conditional_losses_29531

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€   :::W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs
Д
~
)__inference_conv2d_39_layer_call_fn_29482

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_39_layer_call_and_return_conditional_losses_285092
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Џ
d
H__inference_activation_54_layer_call_and_return_conditional_losses_28650

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:€€€€€€€€€А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Њ
`
D__inference_flatten_6_layer_call_and_return_conditional_losses_28705

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А 2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А 2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
†
Б
,__inference_sequential_7_layer_call_fn_29142
conv2d_38_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identityИҐStatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCallconv2d_38_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_7_layer_call_and_return_conditional_losses_291032
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:€€€€€€€€€АА
)
_user_specified_nameconv2d_38_input
Й
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_28790

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ф2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€ф:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
Е
ш
,__inference_sequential_7_layer_call_fn_29393

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identityИҐStatefulPartitionedCallќ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_7_layer_call_and_return_conditional_losses_289962
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
•
ђ
D__inference_conv2d_40_layer_call_and_return_conditional_losses_28549

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@@:::W S
/
_output_shapes
:€€€€€€€€€@@
 
_user_specified_nameinputs
д~
с
__inference__traced_save_29945
file_prefix/
+savev2_conv2d_38_kernel_read_readvariableop-
)savev2_conv2d_38_bias_read_readvariableop/
+savev2_conv2d_39_kernel_read_readvariableop-
)savev2_conv2d_39_bias_read_readvariableop/
+savev2_conv2d_40_kernel_read_readvariableop-
)savev2_conv2d_40_bias_read_readvariableop/
+savev2_conv2d_41_kernel_read_readvariableop-
)savev2_conv2d_41_bias_read_readvariableop/
+savev2_conv2d_42_kernel_read_readvariableop-
)savev2_conv2d_42_bias_read_readvariableop/
+savev2_conv2d_43_kernel_read_readvariableop-
)savev2_conv2d_43_bias_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_38_kernel_m_read_readvariableop4
0savev2_adam_conv2d_38_bias_m_read_readvariableop6
2savev2_adam_conv2d_39_kernel_m_read_readvariableop4
0savev2_adam_conv2d_39_bias_m_read_readvariableop6
2savev2_adam_conv2d_40_kernel_m_read_readvariableop4
0savev2_adam_conv2d_40_bias_m_read_readvariableop6
2savev2_adam_conv2d_41_kernel_m_read_readvariableop4
0savev2_adam_conv2d_41_bias_m_read_readvariableop6
2savev2_adam_conv2d_42_kernel_m_read_readvariableop4
0savev2_adam_conv2d_42_bias_m_read_readvariableop6
2savev2_adam_conv2d_43_kernel_m_read_readvariableop4
0savev2_adam_conv2d_43_bias_m_read_readvariableop5
1savev2_adam_dense_14_kernel_m_read_readvariableop3
/savev2_adam_dense_14_bias_m_read_readvariableop5
1savev2_adam_dense_15_kernel_m_read_readvariableop3
/savev2_adam_dense_15_bias_m_read_readvariableop5
1savev2_adam_dense_16_kernel_m_read_readvariableop3
/savev2_adam_dense_16_bias_m_read_readvariableop6
2savev2_adam_conv2d_38_kernel_v_read_readvariableop4
0savev2_adam_conv2d_38_bias_v_read_readvariableop6
2savev2_adam_conv2d_39_kernel_v_read_readvariableop4
0savev2_adam_conv2d_39_bias_v_read_readvariableop6
2savev2_adam_conv2d_40_kernel_v_read_readvariableop4
0savev2_adam_conv2d_40_bias_v_read_readvariableop6
2savev2_adam_conv2d_41_kernel_v_read_readvariableop4
0savev2_adam_conv2d_41_bias_v_read_readvariableop6
2savev2_adam_conv2d_42_kernel_v_read_readvariableop4
0savev2_adam_conv2d_42_bias_v_read_readvariableop6
2savev2_adam_conv2d_43_kernel_v_read_readvariableop4
0savev2_adam_conv2d_43_bias_v_read_readvariableop5
1savev2_adam_dense_14_kernel_v_read_readvariableop3
/savev2_adam_dense_14_bias_v_read_readvariableop5
1savev2_adam_dense_15_kernel_v_read_readvariableop3
/savev2_adam_dense_15_bias_v_read_readvariableop5
1savev2_adam_dense_16_kernel_v_read_readvariableop3
/savev2_adam_dense_16_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_ea3964c9c34143c1800258ccf408823a/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameи#
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*ъ"
valueр"Bн"@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЛ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*Х
valueЛBИ@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesГ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_38_kernel_read_readvariableop)savev2_conv2d_38_bias_read_readvariableop+savev2_conv2d_39_kernel_read_readvariableop)savev2_conv2d_39_bias_read_readvariableop+savev2_conv2d_40_kernel_read_readvariableop)savev2_conv2d_40_bias_read_readvariableop+savev2_conv2d_41_kernel_read_readvariableop)savev2_conv2d_41_bias_read_readvariableop+savev2_conv2d_42_kernel_read_readvariableop)savev2_conv2d_42_bias_read_readvariableop+savev2_conv2d_43_kernel_read_readvariableop)savev2_conv2d_43_bias_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_38_kernel_m_read_readvariableop0savev2_adam_conv2d_38_bias_m_read_readvariableop2savev2_adam_conv2d_39_kernel_m_read_readvariableop0savev2_adam_conv2d_39_bias_m_read_readvariableop2savev2_adam_conv2d_40_kernel_m_read_readvariableop0savev2_adam_conv2d_40_bias_m_read_readvariableop2savev2_adam_conv2d_41_kernel_m_read_readvariableop0savev2_adam_conv2d_41_bias_m_read_readvariableop2savev2_adam_conv2d_42_kernel_m_read_readvariableop0savev2_adam_conv2d_42_bias_m_read_readvariableop2savev2_adam_conv2d_43_kernel_m_read_readvariableop0savev2_adam_conv2d_43_bias_m_read_readvariableop1savev2_adam_dense_14_kernel_m_read_readvariableop/savev2_adam_dense_14_bias_m_read_readvariableop1savev2_adam_dense_15_kernel_m_read_readvariableop/savev2_adam_dense_15_bias_m_read_readvariableop1savev2_adam_dense_16_kernel_m_read_readvariableop/savev2_adam_dense_16_bias_m_read_readvariableop2savev2_adam_conv2d_38_kernel_v_read_readvariableop0savev2_adam_conv2d_38_bias_v_read_readvariableop2savev2_adam_conv2d_39_kernel_v_read_readvariableop0savev2_adam_conv2d_39_bias_v_read_readvariableop2savev2_adam_conv2d_40_kernel_v_read_readvariableop0savev2_adam_conv2d_40_bias_v_read_readvariableop2savev2_adam_conv2d_41_kernel_v_read_readvariableop0savev2_adam_conv2d_41_bias_v_read_readvariableop2savev2_adam_conv2d_42_kernel_v_read_readvariableop0savev2_adam_conv2d_42_bias_v_read_readvariableop2savev2_adam_conv2d_43_kernel_v_read_readvariableop0savev2_adam_conv2d_43_bias_v_read_readvariableop1savev2_adam_dense_14_kernel_v_read_readvariableop/savev2_adam_dense_14_bias_v_read_readvariableop1savev2_adam_dense_15_kernel_v_read_readvariableop/savev2_adam_dense_15_bias_v_read_readvariableop1savev2_adam_dense_16_kernel_v_read_readvariableop/savev2_adam_dense_16_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *N
dtypesD
B2@	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*П
_input_shapesэ
ъ: ::::: : : @:@:@А:А:АА:А:
А и:и:
иф:ф:	ф:: : : : : : : : : ::::: : : @:@:@А:А:АА:А:
А и:и:
иф:ф:	ф:::::: : : @:@:@А:А:АА:А:
А и:и:
иф:ф:	ф:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-	)
'
_output_shapes
:@А:!


_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:&"
 
_output_shapes
:
А и:!

_output_shapes	
:и:&"
 
_output_shapes
:
иф:!

_output_shapes	
:ф:%!

_output_shapes
:	ф: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
: : !

_output_shapes
: :,"(
&
_output_shapes
: @: #

_output_shapes
:@:-$)
'
_output_shapes
:@А:!%

_output_shapes	
:А:.&*
(
_output_shapes
:АА:!'

_output_shapes	
:А:&("
 
_output_shapes
:
А и:!)

_output_shapes	
:и:&*"
 
_output_shapes
:
иф:!+

_output_shapes	
:ф:%,!

_output_shapes
:	ф: -

_output_shapes
::,.(
&
_output_shapes
:: /

_output_shapes
::,0(
&
_output_shapes
:: 1

_output_shapes
::,2(
&
_output_shapes
: : 3

_output_shapes
: :,4(
&
_output_shapes
: @: 5

_output_shapes
:@:-6)
'
_output_shapes
:@А:!7

_output_shapes	
:А:.8*
(
_output_shapes
:АА:!9

_output_shapes	
:А:&:"
 
_output_shapes
:
А и:!;

_output_shapes	
:и:&<"
 
_output_shapes
:
иф:!=

_output_shapes	
:ф:%>!

_output_shapes
:	ф: ?

_output_shapes
::@

_output_shapes
: 
Б
g
K__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_28413

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Џ
d
H__inference_activation_55_layer_call_and_return_conditional_losses_28690

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:€€€€€€€€€А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
‘
Ђ
C__inference_dense_14_layer_call_and_return_conditional_losses_28723

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А и*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€и2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:и*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€и2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€и2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А :::P L
(
_output_shapes
:€€€€€€€€€А 
 
_user_specified_nameinputs
ё
}
(__inference_dense_15_layer_call_fn_29667

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_287622
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€и::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
≠
L
0__inference_max_pooling2d_39_layer_call_fn_28407

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_284012
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Й
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_29679

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€ф2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€ф:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
™
ђ
D__inference_conv2d_42_layer_call_and_return_conditional_losses_28629

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@:::W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
≠
L
0__inference_max_pooling2d_42_layer_call_fn_28443

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_284372
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
¶И
з 
!__inference__traced_restore_30144
file_prefix%
!assignvariableop_conv2d_38_kernel%
!assignvariableop_1_conv2d_38_bias'
#assignvariableop_2_conv2d_39_kernel%
!assignvariableop_3_conv2d_39_bias'
#assignvariableop_4_conv2d_40_kernel%
!assignvariableop_5_conv2d_40_bias'
#assignvariableop_6_conv2d_41_kernel%
!assignvariableop_7_conv2d_41_bias'
#assignvariableop_8_conv2d_42_kernel%
!assignvariableop_9_conv2d_42_bias(
$assignvariableop_10_conv2d_43_kernel&
"assignvariableop_11_conv2d_43_bias'
#assignvariableop_12_dense_14_kernel%
!assignvariableop_13_dense_14_bias'
#assignvariableop_14_dense_15_kernel%
!assignvariableop_15_dense_15_bias'
#assignvariableop_16_dense_16_kernel%
!assignvariableop_17_dense_16_bias!
assignvariableop_18_adam_iter#
assignvariableop_19_adam_beta_1#
assignvariableop_20_adam_beta_2"
assignvariableop_21_adam_decay*
&assignvariableop_22_adam_learning_rate
assignvariableop_23_total
assignvariableop_24_count
assignvariableop_25_total_1
assignvariableop_26_count_1/
+assignvariableop_27_adam_conv2d_38_kernel_m-
)assignvariableop_28_adam_conv2d_38_bias_m/
+assignvariableop_29_adam_conv2d_39_kernel_m-
)assignvariableop_30_adam_conv2d_39_bias_m/
+assignvariableop_31_adam_conv2d_40_kernel_m-
)assignvariableop_32_adam_conv2d_40_bias_m/
+assignvariableop_33_adam_conv2d_41_kernel_m-
)assignvariableop_34_adam_conv2d_41_bias_m/
+assignvariableop_35_adam_conv2d_42_kernel_m-
)assignvariableop_36_adam_conv2d_42_bias_m/
+assignvariableop_37_adam_conv2d_43_kernel_m-
)assignvariableop_38_adam_conv2d_43_bias_m.
*assignvariableop_39_adam_dense_14_kernel_m,
(assignvariableop_40_adam_dense_14_bias_m.
*assignvariableop_41_adam_dense_15_kernel_m,
(assignvariableop_42_adam_dense_15_bias_m.
*assignvariableop_43_adam_dense_16_kernel_m,
(assignvariableop_44_adam_dense_16_bias_m/
+assignvariableop_45_adam_conv2d_38_kernel_v-
)assignvariableop_46_adam_conv2d_38_bias_v/
+assignvariableop_47_adam_conv2d_39_kernel_v-
)assignvariableop_48_adam_conv2d_39_bias_v/
+assignvariableop_49_adam_conv2d_40_kernel_v-
)assignvariableop_50_adam_conv2d_40_bias_v/
+assignvariableop_51_adam_conv2d_41_kernel_v-
)assignvariableop_52_adam_conv2d_41_bias_v/
+assignvariableop_53_adam_conv2d_42_kernel_v-
)assignvariableop_54_adam_conv2d_42_bias_v/
+assignvariableop_55_adam_conv2d_43_kernel_v-
)assignvariableop_56_adam_conv2d_43_bias_v.
*assignvariableop_57_adam_dense_14_kernel_v,
(assignvariableop_58_adam_dense_14_bias_v.
*assignvariableop_59_adam_dense_15_kernel_v,
(assignvariableop_60_adam_dense_15_bias_v.
*assignvariableop_61_adam_dense_16_kernel_v,
(assignvariableop_62_adam_dense_16_bias_v
identity_64ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9о#
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*ъ"
valueр"Bн"@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesС
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*Х
valueЛBИ@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesо
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ц
_output_shapesГ
А::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*N
dtypesD
B2@	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity†
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_38_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¶
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_38_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2®
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_39_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¶
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_39_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4®
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_40_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¶
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_40_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6®
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_41_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¶
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_41_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8®
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_42_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¶
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_42_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ђ
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_43_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11™
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_43_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ђ
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_14_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_14_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ђ
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_15_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_15_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ђ
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_16_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17©
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_16_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_18•
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19І
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20І
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¶
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_decayIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ѓ
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_learning_rateIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23°
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24°
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25£
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26£
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27≥
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_38_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28±
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_38_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29≥
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_39_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30±
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_39_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31≥
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_40_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32±
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_40_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33≥
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_41_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34±
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_41_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35≥
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_42_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36±
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_42_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37≥
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_43_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38±
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_43_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39≤
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_14_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40∞
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_14_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41≤
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_15_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42∞
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_15_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43≤
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_16_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44∞
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_16_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45≥
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_38_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46±
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_38_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47≥
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_39_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48±
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_39_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49≥
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv2d_40_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50±
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv2d_40_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51≥
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv2d_41_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52±
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv2d_41_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53≥
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv2d_42_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54±
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_conv2d_42_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55≥
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv2d_43_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56±
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv2d_43_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57≤
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_dense_14_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58∞
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_dense_14_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59≤
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_dense_15_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60∞
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_dense_15_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61≤
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_dense_16_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62∞
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_dense_16_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_629
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp»
Identity_63Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_63ї
Identity_64IdentityIdentity_63:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_64"#
identity_64Identity_64:output:0*У
_input_shapesБ
ю: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ї
d
H__inference_activation_57_layer_call_and_return_conditional_losses_29699

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:€€€€€€€€€ф2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€ф:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
р

ш
#__inference_signature_wrapper_29193
conv2d_38_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identityИҐStatefulPartitionedCall∞
StatefulPartitionedCallStatefulPartitionedCallconv2d_38_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_283832
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:€€€€€€€€€АА
)
_user_specified_nameconv2d_38_input
÷
d
H__inference_activation_53_layer_call_and_return_conditional_losses_28610

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€  @2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€  @2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€  @:W S
/
_output_shapes
:€€€€€€€€€  @
 
_user_specified_nameinputs
¬
I
-__inference_activation_50_layer_call_fn_29463

inputs
identity–
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_50_layer_call_and_return_conditional_losses_284902
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
ь
~
)__inference_conv2d_41_layer_call_fn_29540

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_285892
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€  @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€   ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs
Б
g
K__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_28401

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
†
Б
,__inference_sequential_7_layer_call_fn_29035
conv2d_38_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identityИҐStatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCallconv2d_38_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_7_layer_call_and_return_conditional_losses_289962
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:€€€€€€€€€АА
)
_user_specified_nameconv2d_38_input
Д
~
)__inference_conv2d_38_layer_call_fn_29453

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_38_layer_call_and_return_conditional_losses_284692
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Ї
d
H__inference_activation_56_layer_call_and_return_conditional_losses_28744

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:€€€€€€€€€и2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€и2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€и:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
≠
L
0__inference_max_pooling2d_40_layer_call_fn_28419

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_284132
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ѓ
ђ
D__inference_conv2d_38_layer_call_and_return_conditional_losses_29444

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp•
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА2	
BiasAddn
IdentityIdentityBiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА:::Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
ё
d
H__inference_activation_51_layer_call_and_return_conditional_losses_29487

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:€€€€€€€€€АА2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Б
g
K__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_28425

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
иd
Ы
G__inference_sequential_7_layer_call_and_return_conditional_losses_29103

inputs
conv2d_38_29040
conv2d_38_29042
conv2d_39_29047
conv2d_39_29049
conv2d_40_29054
conv2d_40_29056
conv2d_41_29061
conv2d_41_29063
conv2d_42_29068
conv2d_42_29070
conv2d_43_29075
conv2d_43_29077
dense_14_29083
dense_14_29085
dense_15_29089
dense_15_29091
dense_16_29096
dense_16_29098
identityИҐ!conv2d_38/StatefulPartitionedCallҐ!conv2d_39/StatefulPartitionedCallҐ!conv2d_40/StatefulPartitionedCallҐ!conv2d_41/StatefulPartitionedCallҐ!conv2d_42/StatefulPartitionedCallҐ!conv2d_43/StatefulPartitionedCallҐ dense_14/StatefulPartitionedCallҐ dense_15/StatefulPartitionedCallҐ dense_16/StatefulPartitionedCall†
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_38_29040conv2d_38_29042*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_38_layer_call_and_return_conditional_losses_284692#
!conv2d_38/StatefulPartitionedCallР
activation_50/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_50_layer_call_and_return_conditional_losses_284902
activation_50/PartitionedCallХ
 max_pooling2d_38/PartitionedCallPartitionedCall&activation_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_283892"
 max_pooling2d_38/PartitionedCall√
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_38/PartitionedCall:output:0conv2d_39_29047conv2d_39_29049*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_39_layer_call_and_return_conditional_losses_285092#
!conv2d_39/StatefulPartitionedCallР
activation_51/PartitionedCallPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_51_layer_call_and_return_conditional_losses_285302
activation_51/PartitionedCallУ
 max_pooling2d_39/PartitionedCallPartitionedCall&activation_51/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_284012"
 max_pooling2d_39/PartitionedCallЅ
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_39/PartitionedCall:output:0conv2d_40_29054conv2d_40_29056*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_40_layer_call_and_return_conditional_losses_285492#
!conv2d_40/StatefulPartitionedCallО
activation_52/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_52_layer_call_and_return_conditional_losses_285702
activation_52/PartitionedCallУ
 max_pooling2d_40/PartitionedCallPartitionedCall&activation_52/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_284132"
 max_pooling2d_40/PartitionedCallЅ
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_40/PartitionedCall:output:0conv2d_41_29061conv2d_41_29063*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_285892#
!conv2d_41/StatefulPartitionedCallО
activation_53/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_53_layer_call_and_return_conditional_losses_286102
activation_53/PartitionedCallУ
 max_pooling2d_41/PartitionedCallPartitionedCall&activation_53/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_284252"
 max_pooling2d_41/PartitionedCall¬
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_41/PartitionedCall:output:0conv2d_42_29068conv2d_42_29070*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_42_layer_call_and_return_conditional_losses_286292#
!conv2d_42/StatefulPartitionedCallП
activation_54/PartitionedCallPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_54_layer_call_and_return_conditional_losses_286502
activation_54/PartitionedCallФ
 max_pooling2d_42/PartitionedCallPartitionedCall&activation_54/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_284372"
 max_pooling2d_42/PartitionedCall¬
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_42/PartitionedCall:output:0conv2d_43_29075conv2d_43_29077*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_43_layer_call_and_return_conditional_losses_286692#
!conv2d_43/StatefulPartitionedCallП
activation_55/PartitionedCallPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_55_layer_call_and_return_conditional_losses_286902
activation_55/PartitionedCallФ
 max_pooling2d_43/PartitionedCallPartitionedCall&activation_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_43_layer_call_and_return_conditional_losses_284492"
 max_pooling2d_43/PartitionedCallъ
flatten_6/PartitionedCallPartitionedCall)max_pooling2d_43/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_287052
flatten_6/PartitionedCallЃ
 dense_14/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_14_29083dense_14_29085*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_287232"
 dense_14/StatefulPartitionedCallЖ
activation_56/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_56_layer_call_and_return_conditional_losses_287442
activation_56/PartitionedCall≤
 dense_15/StatefulPartitionedCallStatefulPartitionedCall&activation_56/PartitionedCall:output:0dense_15_29089dense_15_29091*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_287622"
 dense_15/StatefulPartitionedCallъ
dropout_4/PartitionedCallPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_287952
dropout_4/PartitionedCall€
activation_57/PartitionedCallPartitionedCall"dropout_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_57_layer_call_and_return_conditional_losses_288132
activation_57/PartitionedCall±
 dense_16/StatefulPartitionedCallStatefulPartitionedCall&activation_57/PartitionedCall:output:0dense_16_29096dense_16_29098*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_288312"
 dense_16/StatefulPartitionedCallЕ
activation_58/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_58_layer_call_and_return_conditional_losses_288522
activation_58/PartitionedCallї
IdentityIdentity&activation_58/PartitionedCall:output:0"^conv2d_38/StatefulPartitionedCall"^conv2d_39/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА::::::::::::::::::2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Ї
I
-__inference_activation_52_layer_call_fn_29521

inputs
identityќ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_52_layer_call_and_return_conditional_losses_285702
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@@ :W S
/
_output_shapes
:€€€€€€€€€@@ 
 
_user_specified_nameinputs
¶
E
)__inference_flatten_6_layer_call_fn_29619

inputs
identity√
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_287052
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А 2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
¬
I
-__inference_activation_51_layer_call_fn_29492

inputs
identity–
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_51_layer_call_and_return_conditional_losses_285302
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
‘
Ђ
C__inference_dense_15_layer_call_and_return_conditional_losses_29658

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
иф*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ф*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ф2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€и:::P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
Ї
I
-__inference_activation_53_layer_call_fn_29550

inputs
identityќ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_53_layer_call_and_return_conditional_losses_286102
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€  @2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€  @:W S
/
_output_shapes
:€€€€€€€€€  @
 
_user_specified_nameinputs
ё
}
(__inference_dense_14_layer_call_fn_29638

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_287232
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€и2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А 
 
_user_specified_nameinputs
Њ
`
D__inference_flatten_6_layer_call_and_return_conditional_losses_29614

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А 2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А 2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ґ
b
)__inference_dropout_4_layer_call_fn_29689

inputs
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_287902
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€ф22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
Гe
§
G__inference_sequential_7_layer_call_and_return_conditional_losses_28927
conv2d_38_input
conv2d_38_28864
conv2d_38_28866
conv2d_39_28871
conv2d_39_28873
conv2d_40_28878
conv2d_40_28880
conv2d_41_28885
conv2d_41_28887
conv2d_42_28892
conv2d_42_28894
conv2d_43_28899
conv2d_43_28901
dense_14_28907
dense_14_28909
dense_15_28913
dense_15_28915
dense_16_28920
dense_16_28922
identityИҐ!conv2d_38/StatefulPartitionedCallҐ!conv2d_39/StatefulPartitionedCallҐ!conv2d_40/StatefulPartitionedCallҐ!conv2d_41/StatefulPartitionedCallҐ!conv2d_42/StatefulPartitionedCallҐ!conv2d_43/StatefulPartitionedCallҐ dense_14/StatefulPartitionedCallҐ dense_15/StatefulPartitionedCallҐ dense_16/StatefulPartitionedCall©
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCallconv2d_38_inputconv2d_38_28864conv2d_38_28866*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_38_layer_call_and_return_conditional_losses_284692#
!conv2d_38/StatefulPartitionedCallР
activation_50/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_50_layer_call_and_return_conditional_losses_284902
activation_50/PartitionedCallХ
 max_pooling2d_38/PartitionedCallPartitionedCall&activation_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_283892"
 max_pooling2d_38/PartitionedCall√
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_38/PartitionedCall:output:0conv2d_39_28871conv2d_39_28873*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_39_layer_call_and_return_conditional_losses_285092#
!conv2d_39/StatefulPartitionedCallР
activation_51/PartitionedCallPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_51_layer_call_and_return_conditional_losses_285302
activation_51/PartitionedCallУ
 max_pooling2d_39/PartitionedCallPartitionedCall&activation_51/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_284012"
 max_pooling2d_39/PartitionedCallЅ
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_39/PartitionedCall:output:0conv2d_40_28878conv2d_40_28880*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_40_layer_call_and_return_conditional_losses_285492#
!conv2d_40/StatefulPartitionedCallО
activation_52/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_52_layer_call_and_return_conditional_losses_285702
activation_52/PartitionedCallУ
 max_pooling2d_40/PartitionedCallPartitionedCall&activation_52/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_284132"
 max_pooling2d_40/PartitionedCallЅ
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_40/PartitionedCall:output:0conv2d_41_28885conv2d_41_28887*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_285892#
!conv2d_41/StatefulPartitionedCallО
activation_53/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_53_layer_call_and_return_conditional_losses_286102
activation_53/PartitionedCallУ
 max_pooling2d_41/PartitionedCallPartitionedCall&activation_53/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_284252"
 max_pooling2d_41/PartitionedCall¬
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_41/PartitionedCall:output:0conv2d_42_28892conv2d_42_28894*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_42_layer_call_and_return_conditional_losses_286292#
!conv2d_42/StatefulPartitionedCallП
activation_54/PartitionedCallPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_54_layer_call_and_return_conditional_losses_286502
activation_54/PartitionedCallФ
 max_pooling2d_42/PartitionedCallPartitionedCall&activation_54/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_284372"
 max_pooling2d_42/PartitionedCall¬
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_42/PartitionedCall:output:0conv2d_43_28899conv2d_43_28901*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_43_layer_call_and_return_conditional_losses_286692#
!conv2d_43/StatefulPartitionedCallП
activation_55/PartitionedCallPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_55_layer_call_and_return_conditional_losses_286902
activation_55/PartitionedCallФ
 max_pooling2d_43/PartitionedCallPartitionedCall&activation_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_43_layer_call_and_return_conditional_losses_284492"
 max_pooling2d_43/PartitionedCallъ
flatten_6/PartitionedCallPartitionedCall)max_pooling2d_43/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_287052
flatten_6/PartitionedCallЃ
 dense_14/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_14_28907dense_14_28909*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_287232"
 dense_14/StatefulPartitionedCallЖ
activation_56/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_56_layer_call_and_return_conditional_losses_287442
activation_56/PartitionedCall≤
 dense_15/StatefulPartitionedCallStatefulPartitionedCall&activation_56/PartitionedCall:output:0dense_15_28913dense_15_28915*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_287622"
 dense_15/StatefulPartitionedCallъ
dropout_4/PartitionedCallPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_287952
dropout_4/PartitionedCall€
activation_57/PartitionedCallPartitionedCall"dropout_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_57_layer_call_and_return_conditional_losses_288132
activation_57/PartitionedCall±
 dense_16/StatefulPartitionedCallStatefulPartitionedCall&activation_57/PartitionedCall:output:0dense_16_28920dense_16_28922*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_288312"
 dense_16/StatefulPartitionedCallЕ
activation_58/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_58_layer_call_and_return_conditional_losses_288522
activation_58/PartitionedCallї
IdentityIdentity&activation_58/PartitionedCall:output:0"^conv2d_38/StatefulPartitionedCall"^conv2d_39/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА::::::::::::::::::2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall:b ^
1
_output_shapes
:€€€€€€€€€АА
)
_user_specified_nameconv2d_38_input
ѓ
ђ
D__inference_conv2d_39_layer_call_and_return_conditional_losses_29473

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp•
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА2	
BiasAddn
IdentityIdentityBiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА:::Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Б
g
K__inference_max_pooling2d_43_layer_call_and_return_conditional_losses_28449

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ћ
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_29684

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:€€€€€€€€€ф:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
ѓ
ђ
D__inference_conv2d_38_layer_call_and_return_conditional_losses_28469

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp•
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА2	
BiasAddn
IdentityIdentityBiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€АА:::Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Б
g
K__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_28389

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Чh
с
 __inference__wrapped_model_28383
conv2d_38_input9
5sequential_7_conv2d_38_conv2d_readvariableop_resource:
6sequential_7_conv2d_38_biasadd_readvariableop_resource9
5sequential_7_conv2d_39_conv2d_readvariableop_resource:
6sequential_7_conv2d_39_biasadd_readvariableop_resource9
5sequential_7_conv2d_40_conv2d_readvariableop_resource:
6sequential_7_conv2d_40_biasadd_readvariableop_resource9
5sequential_7_conv2d_41_conv2d_readvariableop_resource:
6sequential_7_conv2d_41_biasadd_readvariableop_resource9
5sequential_7_conv2d_42_conv2d_readvariableop_resource:
6sequential_7_conv2d_42_biasadd_readvariableop_resource9
5sequential_7_conv2d_43_conv2d_readvariableop_resource:
6sequential_7_conv2d_43_biasadd_readvariableop_resource8
4sequential_7_dense_14_matmul_readvariableop_resource9
5sequential_7_dense_14_biasadd_readvariableop_resource8
4sequential_7_dense_15_matmul_readvariableop_resource9
5sequential_7_dense_15_biasadd_readvariableop_resource8
4sequential_7_dense_16_matmul_readvariableop_resource9
5sequential_7_dense_16_biasadd_readvariableop_resource
identityИЏ
,sequential_7/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_38_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,sequential_7/conv2d_38/Conv2D/ReadVariableOpу
sequential_7/conv2d_38/Conv2DConv2Dconv2d_38_input4sequential_7/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingSAME*
strides
2
sequential_7/conv2d_38/Conv2D—
-sequential_7/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_7/conv2d_38/BiasAdd/ReadVariableOpж
sequential_7/conv2d_38/BiasAddBiasAdd&sequential_7/conv2d_38/Conv2D:output:05sequential_7/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА2 
sequential_7/conv2d_38/BiasAddѓ
sequential_7/activation_50/ReluRelu'sequential_7/conv2d_38/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2!
sequential_7/activation_50/Reluч
%sequential_7/max_pooling2d_38/MaxPoolMaxPool-sequential_7/activation_50/Relu:activations:0*1
_output_shapes
:€€€€€€€€€АА*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_38/MaxPoolЏ
,sequential_7/conv2d_39/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_39_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,sequential_7/conv2d_39/Conv2D/ReadVariableOpТ
sequential_7/conv2d_39/Conv2DConv2D.sequential_7/max_pooling2d_38/MaxPool:output:04sequential_7/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingSAME*
strides
2
sequential_7/conv2d_39/Conv2D—
-sequential_7/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_7/conv2d_39/BiasAdd/ReadVariableOpж
sequential_7/conv2d_39/BiasAddBiasAdd&sequential_7/conv2d_39/Conv2D:output:05sequential_7/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА2 
sequential_7/conv2d_39/BiasAddѓ
sequential_7/activation_51/ReluRelu'sequential_7/conv2d_39/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2!
sequential_7/activation_51/Reluх
%sequential_7/max_pooling2d_39/MaxPoolMaxPool-sequential_7/activation_51/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@@*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_39/MaxPoolЏ
,sequential_7/conv2d_40/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_40/Conv2D/ReadVariableOpР
sequential_7/conv2d_40/Conv2DConv2D.sequential_7/max_pooling2d_39/MaxPool:output:04sequential_7/conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
paddingSAME*
strides
2
sequential_7/conv2d_40/Conv2D—
-sequential_7/conv2d_40/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_40/BiasAdd/ReadVariableOpд
sequential_7/conv2d_40/BiasAddBiasAdd&sequential_7/conv2d_40/Conv2D:output:05sequential_7/conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2 
sequential_7/conv2d_40/BiasAdd≠
sequential_7/activation_52/ReluRelu'sequential_7/conv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2!
sequential_7/activation_52/Reluх
%sequential_7/max_pooling2d_40/MaxPoolMaxPool-sequential_7/activation_52/Relu:activations:0*/
_output_shapes
:€€€€€€€€€   *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_40/MaxPoolЏ
,sequential_7/conv2d_41/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02.
,sequential_7/conv2d_41/Conv2D/ReadVariableOpР
sequential_7/conv2d_41/Conv2DConv2D.sequential_7/max_pooling2d_40/MaxPool:output:04sequential_7/conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingSAME*
strides
2
sequential_7/conv2d_41/Conv2D—
-sequential_7/conv2d_41/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_7/conv2d_41/BiasAdd/ReadVariableOpд
sequential_7/conv2d_41/BiasAddBiasAdd&sequential_7/conv2d_41/Conv2D:output:05sequential_7/conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @2 
sequential_7/conv2d_41/BiasAdd≠
sequential_7/activation_53/ReluRelu'sequential_7/conv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€  @2!
sequential_7/activation_53/Reluх
%sequential_7/max_pooling2d_41/MaxPoolMaxPool-sequential_7/activation_53/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_41/MaxPoolџ
,sequential_7/conv2d_42/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_42_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02.
,sequential_7/conv2d_42/Conv2D/ReadVariableOpС
sequential_7/conv2d_42/Conv2DConv2D.sequential_7/max_pooling2d_41/MaxPool:output:04sequential_7/conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential_7/conv2d_42/Conv2D“
-sequential_7/conv2d_42/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_42_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_42/BiasAdd/ReadVariableOpе
sequential_7/conv2d_42/BiasAddBiasAdd&sequential_7/conv2d_42/Conv2D:output:05sequential_7/conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2 
sequential_7/conv2d_42/BiasAddЃ
sequential_7/activation_54/ReluRelu'sequential_7/conv2d_42/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2!
sequential_7/activation_54/Reluц
%sequential_7/max_pooling2d_42/MaxPoolMaxPool-sequential_7/activation_54/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_42/MaxPool№
,sequential_7/conv2d_43/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_43_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_43/Conv2D/ReadVariableOpС
sequential_7/conv2d_43/Conv2DConv2D.sequential_7/max_pooling2d_42/MaxPool:output:04sequential_7/conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential_7/conv2d_43/Conv2D“
-sequential_7/conv2d_43/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_43/BiasAdd/ReadVariableOpе
sequential_7/conv2d_43/BiasAddBiasAdd&sequential_7/conv2d_43/Conv2D:output:05sequential_7/conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2 
sequential_7/conv2d_43/BiasAddЃ
sequential_7/activation_55/ReluRelu'sequential_7/conv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2!
sequential_7/activation_55/Reluц
%sequential_7/max_pooling2d_43/MaxPoolMaxPool-sequential_7/activation_55/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_43/MaxPoolН
sequential_7/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
sequential_7/flatten_6/Const’
sequential_7/flatten_6/ReshapeReshape.sequential_7/max_pooling2d_43/MaxPool:output:0%sequential_7/flatten_6/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А 2 
sequential_7/flatten_6/Reshape—
+sequential_7/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А и*
dtype02-
+sequential_7/dense_14/MatMul/ReadVariableOp„
sequential_7/dense_14/MatMulMatMul'sequential_7/flatten_6/Reshape:output:03sequential_7/dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€и2
sequential_7/dense_14/MatMulѕ
,sequential_7/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_14_biasadd_readvariableop_resource*
_output_shapes	
:и*
dtype02.
,sequential_7/dense_14/BiasAdd/ReadVariableOpЏ
sequential_7/dense_14/BiasAddBiasAdd&sequential_7/dense_14/MatMul:product:04sequential_7/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€и2
sequential_7/dense_14/BiasAdd•
sequential_7/activation_56/ReluRelu&sequential_7/dense_14/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€и2!
sequential_7/activation_56/Relu—
+sequential_7/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_15_matmul_readvariableop_resource* 
_output_shapes
:
иф*
dtype02-
+sequential_7/dense_15/MatMul/ReadVariableOpЁ
sequential_7/dense_15/MatMulMatMul-sequential_7/activation_56/Relu:activations:03sequential_7/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
sequential_7/dense_15/MatMulѕ
,sequential_7/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:ф*
dtype02.
,sequential_7/dense_15/BiasAdd/ReadVariableOpЏ
sequential_7/dense_15/BiasAddBiasAdd&sequential_7/dense_15/MatMul:product:04sequential_7/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
sequential_7/dense_15/BiasAdd©
sequential_7/dropout_4/IdentityIdentity&sequential_7/dense_15/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2!
sequential_7/dropout_4/IdentityІ
sequential_7/activation_57/ReluRelu(sequential_7/dropout_4/Identity:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2!
sequential_7/activation_57/Relu–
+sequential_7/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_16_matmul_readvariableop_resource*
_output_shapes
:	ф*
dtype02-
+sequential_7/dense_16/MatMul/ReadVariableOp№
sequential_7/dense_16/MatMulMatMul-sequential_7/activation_57/Relu:activations:03sequential_7/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_7/dense_16/MatMulќ
,sequential_7/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_7/dense_16/BiasAdd/ReadVariableOpў
sequential_7/dense_16/BiasAddBiasAdd&sequential_7/dense_16/MatMul:product:04sequential_7/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_7/dense_16/BiasAdd≠
"sequential_7/activation_58/SoftmaxSoftmax&sequential_7/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2$
"sequential_7/activation_58/SoftmaxА
IdentityIdentity,sequential_7/activation_58/Softmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА:::::::::::::::::::b ^
1
_output_shapes
:€€€€€€€€€АА
)
_user_specified_nameconv2d_38_input
Ї
d
H__inference_activation_56_layer_call_and_return_conditional_losses_29643

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:€€€€€€€€€и2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€и2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€и:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
≠
L
0__inference_max_pooling2d_41_layer_call_fn_28431

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_284252
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
‘
Ђ
C__inference_dense_15_layer_call_and_return_conditional_losses_28762

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
иф*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ф*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ф2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€и:::P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
Ю
I
-__inference_activation_57_layer_call_fn_29704

inputs
identity«
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ф* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_57_layer_call_and_return_conditional_losses_288132
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€ф:P L
(
_output_shapes
:€€€€€€€€€ф
 
_user_specified_nameinputs
ТT
•
G__inference_sequential_7_layer_call_and_return_conditional_losses_29352

inputs,
(conv2d_38_conv2d_readvariableop_resource-
)conv2d_38_biasadd_readvariableop_resource,
(conv2d_39_conv2d_readvariableop_resource-
)conv2d_39_biasadd_readvariableop_resource,
(conv2d_40_conv2d_readvariableop_resource-
)conv2d_40_biasadd_readvariableop_resource,
(conv2d_41_conv2d_readvariableop_resource-
)conv2d_41_biasadd_readvariableop_resource,
(conv2d_42_conv2d_readvariableop_resource-
)conv2d_42_biasadd_readvariableop_resource,
(conv2d_43_conv2d_readvariableop_resource-
)conv2d_43_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource
identityИ≥
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_38/Conv2D/ReadVariableOp√
conv2d_38/Conv2DConv2Dinputs'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingSAME*
strides
2
conv2d_38/Conv2D™
 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_38/BiasAdd/ReadVariableOp≤
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
conv2d_38/BiasAddИ
activation_50/ReluReluconv2d_38/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
activation_50/Relu–
max_pooling2d_38/MaxPoolMaxPool activation_50/Relu:activations:0*1
_output_shapes
:€€€€€€€€€АА*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPool≥
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_39/Conv2D/ReadVariableOpё
conv2d_39/Conv2DConv2D!max_pooling2d_38/MaxPool:output:0'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingSAME*
strides
2
conv2d_39/Conv2D™
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_39/BiasAdd/ReadVariableOp≤
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
conv2d_39/BiasAddИ
activation_51/ReluReluconv2d_39/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
activation_51/Reluќ
max_pooling2d_39/MaxPoolMaxPool activation_51/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_39/MaxPool≥
conv2d_40/Conv2D/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_40/Conv2D/ReadVariableOp№
conv2d_40/Conv2DConv2D!max_pooling2d_39/MaxPool:output:0'conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
paddingSAME*
strides
2
conv2d_40/Conv2D™
 conv2d_40/BiasAdd/ReadVariableOpReadVariableOp)conv2d_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_40/BiasAdd/ReadVariableOp∞
conv2d_40/BiasAddBiasAddconv2d_40/Conv2D:output:0(conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
conv2d_40/BiasAddЖ
activation_52/ReluReluconv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
activation_52/Reluќ
max_pooling2d_40/MaxPoolMaxPool activation_52/Relu:activations:0*/
_output_shapes
:€€€€€€€€€   *
ksize
*
paddingVALID*
strides
2
max_pooling2d_40/MaxPool≥
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_41/Conv2D/ReadVariableOp№
conv2d_41/Conv2DConv2D!max_pooling2d_40/MaxPool:output:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingSAME*
strides
2
conv2d_41/Conv2D™
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_41/BiasAdd/ReadVariableOp∞
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @2
conv2d_41/BiasAddЖ
activation_53/ReluReluconv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€  @2
activation_53/Reluќ
max_pooling2d_41/MaxPoolMaxPool activation_53/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_41/MaxPoolі
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02!
conv2d_42/Conv2D/ReadVariableOpЁ
conv2d_42/Conv2DConv2D!max_pooling2d_41/MaxPool:output:0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_42/Conv2DЂ
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_42/BiasAdd/ReadVariableOp±
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_42/BiasAddЗ
activation_54/ReluReluconv2d_42/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
activation_54/Reluѕ
max_pooling2d_42/MaxPoolMaxPool activation_54/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_42/MaxPoolµ
conv2d_43/Conv2D/ReadVariableOpReadVariableOp(conv2d_43_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_43/Conv2D/ReadVariableOpЁ
conv2d_43/Conv2DConv2D!max_pooling2d_42/MaxPool:output:0'conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_43/Conv2DЂ
 conv2d_43/BiasAdd/ReadVariableOpReadVariableOp)conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_43/BiasAdd/ReadVariableOp±
conv2d_43/BiasAddBiasAddconv2d_43/Conv2D:output:0(conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_43/BiasAddЗ
activation_55/ReluReluconv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
activation_55/Reluѕ
max_pooling2d_43/MaxPoolMaxPool activation_55/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_43/MaxPools
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten_6/Const°
flatten_6/ReshapeReshape!max_pooling2d_43/MaxPool:output:0flatten_6/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А 2
flatten_6/Reshape™
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А и*
dtype02 
dense_14/MatMul/ReadVariableOp£
dense_14/MatMulMatMulflatten_6/Reshape:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€и2
dense_14/MatMul®
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:и*
dtype02!
dense_14/BiasAdd/ReadVariableOp¶
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€и2
dense_14/BiasAdd~
activation_56/ReluReludense_14/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€и2
activation_56/Relu™
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource* 
_output_shapes
:
иф*
dtype02 
dense_15/MatMul/ReadVariableOp©
dense_15/MatMulMatMul activation_56/Relu:activations:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dense_15/MatMul®
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:ф*
dtype02!
dense_15/BiasAdd/ReadVariableOp¶
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dense_15/BiasAddВ
dropout_4/IdentityIdentitydense_15/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
dropout_4/IdentityА
activation_57/ReluReludropout_4/Identity:output:0*
T0*(
_output_shapes
:€€€€€€€€€ф2
activation_57/Relu©
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes
:	ф*
dtype02 
dense_16/MatMul/ReadVariableOp®
dense_16/MatMulMatMul activation_57/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_16/MatMulІ
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_16/BiasAdd/ReadVariableOp•
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_16/BiasAddЖ
activation_58/SoftmaxSoftmaxdense_16/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
activation_58/Softmaxs
IdentityIdentityactivation_58/Softmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:€€€€€€€€€АА:::::::::::::::::::Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Ю
I
-__inference_activation_56_layer_call_fn_29648

inputs
identity«
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_56_layer_call_and_return_conditional_losses_287442
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€и2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€и:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
ю
~
)__inference_conv2d_42_layer_call_fn_29569

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_42_layer_call_and_return_conditional_losses_286292
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs"ЄL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp* 
serving_defaultґ
U
conv2d_38_inputB
!serving_default_conv2d_38_input:0€€€€€€€€€ААA
activation_580
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:‘Ь
№Ц
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer_with_weights-4
layer-12
layer-13
layer-14
layer_with_weights-5
layer-15
layer-16
layer-17
layer-18
layer_with_weights-6
layer-19
layer-20
layer_with_weights-7
layer-21
layer-22
layer-23
layer_with_weights-8
layer-24
layer-25
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api
 
signatures
÷__call__
+„&call_and_return_all_conditional_losses
Ў_default_save_signature"ЅР
_tf_keras_sequential°Р{"class_name": "Sequential", "name": "sequential_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_38_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_38", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 4]}, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_50", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_51", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_40", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_52", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_40", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_53", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_41", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_42", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_54", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_42", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_43", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_55", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_43", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 1000, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_56", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 500, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Activation", "config": {"name": "activation_57", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_58", "trainable": true, "dtype": "float32", "activation": "softmax"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 4]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_38_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_38", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 4]}, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_50", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_51", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_40", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_52", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_40", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_53", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_41", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_42", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_54", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_42", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_43", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_55", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_43", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 1000, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_56", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 500, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Activation", "config": {"name": "activation_57", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_58", "trainable": true, "dtype": "float32", "activation": "softmax"}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 9.999999974752427e-07, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ь


!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api
ў__call__
+Џ&call_and_return_all_conditional_losses"’	
_tf_keras_layerї	{"class_name": "Conv2D", "name": "conv2d_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 4]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_38", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 4]}, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 4]}}
ў
'trainable_variables
(	variables
)regularization_losses
*	keras_api
џ__call__
+№&call_and_return_all_conditional_losses"»
_tf_keras_layerЃ{"class_name": "Activation", "name": "activation_50", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_50", "trainable": true, "dtype": "float32", "activation": "relu"}}
Г
+trainable_variables
,	variables
-regularization_losses
.	keras_api
Ё__call__
+ё&call_and_return_all_conditional_losses"т
_tf_keras_layerЎ{"class_name": "MaxPooling2D", "name": "max_pooling2d_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ш	

/kernel
0bias
1trainable_variables
2	variables
3regularization_losses
4	keras_api
я__call__
+а&call_and_return_all_conditional_losses"—
_tf_keras_layerЈ{"class_name": "Conv2D", "name": "conv2d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_39", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 4]}}
ў
5trainable_variables
6	variables
7regularization_losses
8	keras_api
б__call__
+в&call_and_return_all_conditional_losses"»
_tf_keras_layerЃ{"class_name": "Activation", "name": "activation_51", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_51", "trainable": true, "dtype": "float32", "activation": "relu"}}
Г
9trainable_variables
:	variables
;regularization_losses
<	keras_api
г__call__
+д&call_and_return_all_conditional_losses"т
_tf_keras_layerЎ{"class_name": "MaxPooling2D", "name": "max_pooling2d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ш	

=kernel
>bias
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
е__call__
+ж&call_and_return_all_conditional_losses"—
_tf_keras_layerЈ{"class_name": "Conv2D", "name": "conv2d_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_40", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 16]}}
ў
Ctrainable_variables
D	variables
Eregularization_losses
F	keras_api
з__call__
+и&call_and_return_all_conditional_losses"»
_tf_keras_layerЃ{"class_name": "Activation", "name": "activation_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_52", "trainable": true, "dtype": "float32", "activation": "relu"}}
Г
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
й__call__
+к&call_and_return_all_conditional_losses"т
_tf_keras_layerЎ{"class_name": "MaxPooling2D", "name": "max_pooling2d_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_40", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ш	

Kkernel
Lbias
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
л__call__
+м&call_and_return_all_conditional_losses"—
_tf_keras_layerЈ{"class_name": "Conv2D", "name": "conv2d_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
ў
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
н__call__
+о&call_and_return_all_conditional_losses"»
_tf_keras_layerЃ{"class_name": "Activation", "name": "activation_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_53", "trainable": true, "dtype": "float32", "activation": "relu"}}
Г
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
п__call__
+р&call_and_return_all_conditional_losses"т
_tf_keras_layerЎ{"class_name": "MaxPooling2D", "name": "max_pooling2d_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_41", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
щ	

Ykernel
Zbias
[trainable_variables
\	variables
]regularization_losses
^	keras_api
с__call__
+т&call_and_return_all_conditional_losses"“
_tf_keras_layerЄ{"class_name": "Conv2D", "name": "conv2d_42", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_42", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
ў
_trainable_variables
`	variables
aregularization_losses
b	keras_api
у__call__
+ф&call_and_return_all_conditional_losses"»
_tf_keras_layerЃ{"class_name": "Activation", "name": "activation_54", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_54", "trainable": true, "dtype": "float32", "activation": "relu"}}
Г
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
х__call__
+ц&call_and_return_all_conditional_losses"т
_tf_keras_layerЎ{"class_name": "MaxPooling2D", "name": "max_pooling2d_42", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_42", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
щ	

gkernel
hbias
itrainable_variables
j	variables
kregularization_losses
l	keras_api
ч__call__
+ш&call_and_return_all_conditional_losses"“
_tf_keras_layerЄ{"class_name": "Conv2D", "name": "conv2d_43", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_43", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
ў
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses"»
_tf_keras_layerЃ{"class_name": "Activation", "name": "activation_55", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_55", "trainable": true, "dtype": "float32", "activation": "relu"}}
Г
qtrainable_variables
r	variables
sregularization_losses
t	keras_api
ы__call__
+ь&call_and_return_all_conditional_losses"т
_tf_keras_layerЎ{"class_name": "MaxPooling2D", "name": "max_pooling2d_43", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_43", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
и
utrainable_variables
v	variables
wregularization_losses
x	keras_api
э__call__
+ю&call_and_return_all_conditional_losses"„
_tf_keras_layerљ{"class_name": "Flatten", "name": "flatten_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ь

ykernel
zbias
{trainable_variables
|	variables
}regularization_losses
~	keras_api
€__call__
+А&call_and_return_all_conditional_losses"’
_tf_keras_layerї{"class_name": "Dense", "name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 1000, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4096}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4096]}}
№
trainable_variables
А	variables
Бregularization_losses
В	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"»
_tf_keras_layerЃ{"class_name": "Activation", "name": "activation_56", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_56", "trainable": true, "dtype": "float32", "activation": "relu"}}
Б
Гkernel
	Дbias
Еtrainable_variables
Ж	variables
Зregularization_losses
И	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"‘
_tf_keras_layerЇ{"class_name": "Dense", "name": "dense_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 500, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1000}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000]}}
л
Йtrainable_variables
К	variables
Лregularization_losses
М	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"÷
_tf_keras_layerЉ{"class_name": "Dropout", "name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
Ё
Нtrainable_variables
О	variables
Пregularization_losses
Р	keras_api
З__call__
+И&call_and_return_all_conditional_losses"»
_tf_keras_layerЃ{"class_name": "Activation", "name": "activation_57", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_57", "trainable": true, "dtype": "float32", "activation": "relu"}}
э
Сkernel
	Тbias
Уtrainable_variables
Ф	variables
Хregularization_losses
Ц	keras_api
Й__call__
+К&call_and_return_all_conditional_losses"–
_tf_keras_layerґ{"class_name": "Dense", "name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500]}}
а
Чtrainable_variables
Ш	variables
Щregularization_losses
Ъ	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"Ћ
_tf_keras_layer±{"class_name": "Activation", "name": "activation_58", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_58", "trainable": true, "dtype": "float32", "activation": "softmax"}}
»
	Ыiter
Ьbeta_1
Эbeta_2

Юdecay
Яlearning_rate!m≤"m≥/mі0mµ=mґ>mЈKmЄLmєYmЇZmїgmЉhmљymЊzmњ	Гmј	ДmЅ	Сm¬	Тm√!vƒ"v≈/v∆0v«=v»>v…Kv LvЋYvћZvЌgvќhvѕyv–zv—	Гv“	Дv”	Сv‘	Тv’"
	optimizer
™
!0
"1
/2
03
=4
>5
K6
L7
Y8
Z9
g10
h11
y12
z13
Г14
Д15
С16
Т17"
trackable_list_wrapper
™
!0
"1
/2
03
=4
>5
K6
L7
Y8
Z9
g10
h11
y12
z13
Г14
Д15
С16
Т17"
trackable_list_wrapper
 "
trackable_list_wrapper
”
trainable_variables
	variables
†non_trainable_variables
°metrics
Ґlayers
 £layer_regularization_losses
regularization_losses
§layer_metrics
÷__call__
Ў_default_save_signature
+„&call_and_return_all_conditional_losses
'„"call_and_return_conditional_losses"
_generic_user_object
-
Нserving_default"
signature_map
*:(2conv2d_38/kernel
:2conv2d_38/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
#trainable_variables
$	variables
•metrics
¶non_trainable_variables
Іlayers
 ®layer_regularization_losses
%regularization_losses
©layer_metrics
ў__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
'trainable_variables
(	variables
™metrics
Ђnon_trainable_variables
ђlayers
 ≠layer_regularization_losses
)regularization_losses
Ѓlayer_metrics
џ__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
+trainable_variables
,	variables
ѓmetrics
∞non_trainable_variables
±layers
 ≤layer_regularization_losses
-regularization_losses
≥layer_metrics
Ё__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_39/kernel
:2conv2d_39/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
1trainable_variables
2	variables
іmetrics
µnon_trainable_variables
ґlayers
 Јlayer_regularization_losses
3regularization_losses
Єlayer_metrics
я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
5trainable_variables
6	variables
єmetrics
Їnon_trainable_variables
їlayers
 Љlayer_regularization_losses
7regularization_losses
љlayer_metrics
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
9trainable_variables
:	variables
Њmetrics
њnon_trainable_variables
јlayers
 Ѕlayer_regularization_losses
;regularization_losses
¬layer_metrics
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_40/kernel
: 2conv2d_40/bias
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
?trainable_variables
@	variables
√metrics
ƒnon_trainable_variables
≈layers
 ∆layer_regularization_losses
Aregularization_losses
«layer_metrics
е__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ctrainable_variables
D	variables
»metrics
…non_trainable_variables
 layers
 Ћlayer_regularization_losses
Eregularization_losses
ћlayer_metrics
з__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Gtrainable_variables
H	variables
Ќmetrics
ќnon_trainable_variables
ѕlayers
 –layer_regularization_losses
Iregularization_losses
—layer_metrics
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_41/kernel
:@2conv2d_41/bias
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Mtrainable_variables
N	variables
“metrics
”non_trainable_variables
‘layers
 ’layer_regularization_losses
Oregularization_losses
÷layer_metrics
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Qtrainable_variables
R	variables
„metrics
Ўnon_trainable_variables
ўlayers
 Џlayer_regularization_losses
Sregularization_losses
џlayer_metrics
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Utrainable_variables
V	variables
№metrics
Ёnon_trainable_variables
ёlayers
 яlayer_regularization_losses
Wregularization_losses
аlayer_metrics
п__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
+:)@А2conv2d_42/kernel
:А2conv2d_42/bias
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
[trainable_variables
\	variables
бmetrics
вnon_trainable_variables
гlayers
 дlayer_regularization_losses
]regularization_losses
еlayer_metrics
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
_trainable_variables
`	variables
жmetrics
зnon_trainable_variables
иlayers
 йlayer_regularization_losses
aregularization_losses
кlayer_metrics
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ctrainable_variables
d	variables
лmetrics
мnon_trainable_variables
нlayers
 оlayer_regularization_losses
eregularization_losses
пlayer_metrics
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
,:*АА2conv2d_43/kernel
:А2conv2d_43/bias
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
itrainable_variables
j	variables
рmetrics
сnon_trainable_variables
тlayers
 уlayer_regularization_losses
kregularization_losses
фlayer_metrics
ч__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
mtrainable_variables
n	variables
хmetrics
цnon_trainable_variables
чlayers
 шlayer_regularization_losses
oregularization_losses
щlayer_metrics
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
qtrainable_variables
r	variables
ъmetrics
ыnon_trainable_variables
ьlayers
 эlayer_regularization_losses
sregularization_losses
юlayer_metrics
ы__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
utrainable_variables
v	variables
€metrics
Аnon_trainable_variables
Бlayers
 Вlayer_regularization_losses
wregularization_losses
Гlayer_metrics
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
#:!
А и2dense_14/kernel
:и2dense_14/bias
.
y0
z1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
{trainable_variables
|	variables
Дmetrics
Еnon_trainable_variables
Жlayers
 Зlayer_regularization_losses
}regularization_losses
Иlayer_metrics
€__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ј
trainable_variables
А	variables
Йmetrics
Кnon_trainable_variables
Лlayers
 Мlayer_regularization_losses
Бregularization_losses
Нlayer_metrics
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
#:!
иф2dense_15/kernel
:ф2dense_15/bias
0
Г0
Д1"
trackable_list_wrapper
0
Г0
Д1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Еtrainable_variables
Ж	variables
Оmetrics
Пnon_trainable_variables
Рlayers
 Сlayer_regularization_losses
Зregularization_losses
Тlayer_metrics
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Йtrainable_variables
К	variables
Уmetrics
Фnon_trainable_variables
Хlayers
 Цlayer_regularization_losses
Лregularization_losses
Чlayer_metrics
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Нtrainable_variables
О	variables
Шmetrics
Щnon_trainable_variables
Ъlayers
 Ыlayer_regularization_losses
Пregularization_losses
Ьlayer_metrics
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
": 	ф2dense_16/kernel
:2dense_16/bias
0
С0
Т1"
trackable_list_wrapper
0
С0
Т1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Уtrainable_variables
Ф	variables
Эmetrics
Юnon_trainable_variables
Яlayers
 †layer_regularization_losses
Хregularization_losses
°layer_metrics
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Чtrainable_variables
Ш	variables
Ґmetrics
£non_trainable_variables
§layers
 •layer_regularization_losses
Щregularization_losses
¶layer_metrics
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
0
І0
®1"
trackable_list_wrapper
ж
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
њ

©total

™count
Ђ	variables
ђ	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
Д

≠total

Ѓcount
ѓ
_fn_kwargs
∞	variables
±	keras_api"Є
_tf_keras_metricЭ{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
©0
™1"
trackable_list_wrapper
.
Ђ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
≠0
Ѓ1"
trackable_list_wrapper
.
∞	variables"
_generic_user_object
/:-2Adam/conv2d_38/kernel/m
!:2Adam/conv2d_38/bias/m
/:-2Adam/conv2d_39/kernel/m
!:2Adam/conv2d_39/bias/m
/:- 2Adam/conv2d_40/kernel/m
!: 2Adam/conv2d_40/bias/m
/:- @2Adam/conv2d_41/kernel/m
!:@2Adam/conv2d_41/bias/m
0:.@А2Adam/conv2d_42/kernel/m
": А2Adam/conv2d_42/bias/m
1:/АА2Adam/conv2d_43/kernel/m
": А2Adam/conv2d_43/bias/m
(:&
А и2Adam/dense_14/kernel/m
!:и2Adam/dense_14/bias/m
(:&
иф2Adam/dense_15/kernel/m
!:ф2Adam/dense_15/bias/m
':%	ф2Adam/dense_16/kernel/m
 :2Adam/dense_16/bias/m
/:-2Adam/conv2d_38/kernel/v
!:2Adam/conv2d_38/bias/v
/:-2Adam/conv2d_39/kernel/v
!:2Adam/conv2d_39/bias/v
/:- 2Adam/conv2d_40/kernel/v
!: 2Adam/conv2d_40/bias/v
/:- @2Adam/conv2d_41/kernel/v
!:@2Adam/conv2d_41/bias/v
0:.@А2Adam/conv2d_42/kernel/v
": А2Adam/conv2d_42/bias/v
1:/АА2Adam/conv2d_43/kernel/v
": А2Adam/conv2d_43/bias/v
(:&
А и2Adam/dense_14/kernel/v
!:и2Adam/dense_14/bias/v
(:&
иф2Adam/dense_15/kernel/v
!:ф2Adam/dense_15/bias/v
':%	ф2Adam/dense_16/kernel/v
 :2Adam/dense_16/bias/v
ю2ы
,__inference_sequential_7_layer_call_fn_29434
,__inference_sequential_7_layer_call_fn_29142
,__inference_sequential_7_layer_call_fn_29035
,__inference_sequential_7_layer_call_fn_29393ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
к2з
G__inference_sequential_7_layer_call_and_return_conditional_losses_28927
G__inference_sequential_7_layer_call_and_return_conditional_losses_29276
G__inference_sequential_7_layer_call_and_return_conditional_losses_29352
G__inference_sequential_7_layer_call_and_return_conditional_losses_28861ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
р2н
 __inference__wrapped_model_28383»
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0
conv2d_38_input€€€€€€€€€АА
”2–
)__inference_conv2d_38_layer_call_fn_29453Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_conv2d_38_layer_call_and_return_conditional_losses_29444Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_activation_50_layer_call_fn_29463Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_activation_50_layer_call_and_return_conditional_losses_29458Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ш2Х
0__inference_max_pooling2d_38_layer_call_fn_28395а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
≥2∞
K__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_28389а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
”2–
)__inference_conv2d_39_layer_call_fn_29482Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_conv2d_39_layer_call_and_return_conditional_losses_29473Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_activation_51_layer_call_fn_29492Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_activation_51_layer_call_and_return_conditional_losses_29487Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ш2Х
0__inference_max_pooling2d_39_layer_call_fn_28407а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
≥2∞
K__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_28401а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
”2–
)__inference_conv2d_40_layer_call_fn_29511Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_conv2d_40_layer_call_and_return_conditional_losses_29502Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_activation_52_layer_call_fn_29521Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_activation_52_layer_call_and_return_conditional_losses_29516Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ш2Х
0__inference_max_pooling2d_40_layer_call_fn_28419а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
≥2∞
K__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_28413а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
”2–
)__inference_conv2d_41_layer_call_fn_29540Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_conv2d_41_layer_call_and_return_conditional_losses_29531Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_activation_53_layer_call_fn_29550Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_activation_53_layer_call_and_return_conditional_losses_29545Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ш2Х
0__inference_max_pooling2d_41_layer_call_fn_28431а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
≥2∞
K__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_28425а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
”2–
)__inference_conv2d_42_layer_call_fn_29569Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_conv2d_42_layer_call_and_return_conditional_losses_29560Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_activation_54_layer_call_fn_29579Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_activation_54_layer_call_and_return_conditional_losses_29574Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ш2Х
0__inference_max_pooling2d_42_layer_call_fn_28443а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
≥2∞
K__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_28437а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
”2–
)__inference_conv2d_43_layer_call_fn_29598Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_conv2d_43_layer_call_and_return_conditional_losses_29589Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_activation_55_layer_call_fn_29608Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_activation_55_layer_call_and_return_conditional_losses_29603Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ш2Х
0__inference_max_pooling2d_43_layer_call_fn_28455а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
≥2∞
K__inference_max_pooling2d_43_layer_call_and_return_conditional_losses_28449а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
”2–
)__inference_flatten_6_layer_call_fn_29619Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_flatten_6_layer_call_and_return_conditional_losses_29614Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_14_layer_call_fn_29638Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_14_layer_call_and_return_conditional_losses_29629Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_activation_56_layer_call_fn_29648Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_activation_56_layer_call_and_return_conditional_losses_29643Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_15_layer_call_fn_29667Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_15_layer_call_and_return_conditional_losses_29658Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Р2Н
)__inference_dropout_4_layer_call_fn_29689
)__inference_dropout_4_layer_call_fn_29694і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
∆2√
D__inference_dropout_4_layer_call_and_return_conditional_losses_29684
D__inference_dropout_4_layer_call_and_return_conditional_losses_29679і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
„2‘
-__inference_activation_57_layer_call_fn_29704Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_activation_57_layer_call_and_return_conditional_losses_29699Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_16_layer_call_fn_29723Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_16_layer_call_and_return_conditional_losses_29714Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_activation_58_layer_call_fn_29733Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_activation_58_layer_call_and_return_conditional_losses_29728Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
:B8
#__inference_signature_wrapper_29193conv2d_38_inputј
 __inference__wrapped_model_28383Ы!"/0=>KLYZghyzГДСТBҐ?
8Ґ5
3К0
conv2d_38_input€€€€€€€€€АА
™ "=™:
8
activation_58'К$
activation_58€€€€€€€€€Є
H__inference_activation_50_layer_call_and_return_conditional_losses_29458l9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ Р
-__inference_activation_50_layer_call_fn_29463_9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ ""К€€€€€€€€€ААЄ
H__inference_activation_51_layer_call_and_return_conditional_losses_29487l9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ Р
-__inference_activation_51_layer_call_fn_29492_9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ ""К€€€€€€€€€ААі
H__inference_activation_52_layer_call_and_return_conditional_losses_29516h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@@ 
™ "-Ґ*
#К 
0€€€€€€€€€@@ 
Ъ М
-__inference_activation_52_layer_call_fn_29521[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@@ 
™ " К€€€€€€€€€@@ і
H__inference_activation_53_layer_call_and_return_conditional_losses_29545h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€  @
™ "-Ґ*
#К 
0€€€€€€€€€  @
Ъ М
-__inference_activation_53_layer_call_fn_29550[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€  @
™ " К€€€€€€€€€  @ґ
H__inference_activation_54_layer_call_and_return_conditional_losses_29574j8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ О
-__inference_activation_54_layer_call_fn_29579]8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "!К€€€€€€€€€Аґ
H__inference_activation_55_layer_call_and_return_conditional_losses_29603j8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ О
-__inference_activation_55_layer_call_fn_29608]8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "!К€€€€€€€€€А¶
H__inference_activation_56_layer_call_and_return_conditional_losses_29643Z0Ґ-
&Ґ#
!К
inputs€€€€€€€€€и
™ "&Ґ#
К
0€€€€€€€€€и
Ъ ~
-__inference_activation_56_layer_call_fn_29648M0Ґ-
&Ґ#
!К
inputs€€€€€€€€€и
™ "К€€€€€€€€€и¶
H__inference_activation_57_layer_call_and_return_conditional_losses_29699Z0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ф
™ "&Ґ#
К
0€€€€€€€€€ф
Ъ ~
-__inference_activation_57_layer_call_fn_29704M0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ф
™ "К€€€€€€€€€ф§
H__inference_activation_58_layer_call_and_return_conditional_losses_29728X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ |
-__inference_activation_58_layer_call_fn_29733K/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€Є
D__inference_conv2d_38_layer_call_and_return_conditional_losses_29444p!"9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ Р
)__inference_conv2d_38_layer_call_fn_29453c!"9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ ""К€€€€€€€€€ААЄ
D__inference_conv2d_39_layer_call_and_return_conditional_losses_29473p/09Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ Р
)__inference_conv2d_39_layer_call_fn_29482c/09Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ ""К€€€€€€€€€ААі
D__inference_conv2d_40_layer_call_and_return_conditional_losses_29502l=>7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@@
™ "-Ґ*
#К 
0€€€€€€€€€@@ 
Ъ М
)__inference_conv2d_40_layer_call_fn_29511_=>7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@@
™ " К€€€€€€€€€@@ і
D__inference_conv2d_41_layer_call_and_return_conditional_losses_29531lKL7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€   
™ "-Ґ*
#К 
0€€€€€€€€€  @
Ъ М
)__inference_conv2d_41_layer_call_fn_29540_KL7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€   
™ " К€€€€€€€€€  @µ
D__inference_conv2d_42_layer_call_and_return_conditional_losses_29560mYZ7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ Н
)__inference_conv2d_42_layer_call_fn_29569`YZ7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@
™ "!К€€€€€€€€€Аґ
D__inference_conv2d_43_layer_call_and_return_conditional_losses_29589ngh8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ О
)__inference_conv2d_43_layer_call_fn_29598agh8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "!К€€€€€€€€€А•
C__inference_dense_14_layer_call_and_return_conditional_losses_29629^yz0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А 
™ "&Ґ#
К
0€€€€€€€€€и
Ъ }
(__inference_dense_14_layer_call_fn_29638Qyz0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А 
™ "К€€€€€€€€€иІ
C__inference_dense_15_layer_call_and_return_conditional_losses_29658`ГД0Ґ-
&Ґ#
!К
inputs€€€€€€€€€и
™ "&Ґ#
К
0€€€€€€€€€ф
Ъ 
(__inference_dense_15_layer_call_fn_29667SГД0Ґ-
&Ґ#
!К
inputs€€€€€€€€€и
™ "К€€€€€€€€€ф¶
C__inference_dense_16_layer_call_and_return_conditional_losses_29714_СТ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ф
™ "%Ґ"
К
0€€€€€€€€€
Ъ ~
(__inference_dense_16_layer_call_fn_29723RСТ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ф
™ "К€€€€€€€€€¶
D__inference_dropout_4_layer_call_and_return_conditional_losses_29679^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ф
p
™ "&Ґ#
К
0€€€€€€€€€ф
Ъ ¶
D__inference_dropout_4_layer_call_and_return_conditional_losses_29684^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ф
p 
™ "&Ґ#
К
0€€€€€€€€€ф
Ъ ~
)__inference_dropout_4_layer_call_fn_29689Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ф
p
™ "К€€€€€€€€€ф~
)__inference_dropout_4_layer_call_fn_29694Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€ф
p 
™ "К€€€€€€€€€ф™
D__inference_flatten_6_layer_call_and_return_conditional_losses_29614b8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А 
Ъ В
)__inference_flatten_6_layer_call_fn_29619U8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "К€€€€€€€€€А о
K__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_28389ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∆
0__inference_max_pooling2d_38_layer_call_fn_28395СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€о
K__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_28401ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∆
0__inference_max_pooling2d_39_layer_call_fn_28407СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€о
K__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_28413ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∆
0__inference_max_pooling2d_40_layer_call_fn_28419СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€о
K__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_28425ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∆
0__inference_max_pooling2d_41_layer_call_fn_28431СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€о
K__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_28437ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∆
0__inference_max_pooling2d_42_layer_call_fn_28443СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€о
K__inference_max_pooling2d_43_layer_call_and_return_conditional_losses_28449ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∆
0__inference_max_pooling2d_43_layer_call_fn_28455СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€„
G__inference_sequential_7_layer_call_and_return_conditional_losses_28861Л!"/0=>KLYZghyzГДСТJҐG
@Ґ=
3К0
conv2d_38_input€€€€€€€€€АА
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ „
G__inference_sequential_7_layer_call_and_return_conditional_losses_28927Л!"/0=>KLYZghyzГДСТJҐG
@Ґ=
3К0
conv2d_38_input€€€€€€€€€АА
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ќ
G__inference_sequential_7_layer_call_and_return_conditional_losses_29276В!"/0=>KLYZghyzГДСТAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ќ
G__inference_sequential_7_layer_call_and_return_conditional_losses_29352В!"/0=>KLYZghyzГДСТAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ѓ
,__inference_sequential_7_layer_call_fn_29035~!"/0=>KLYZghyzГДСТJҐG
@Ґ=
3К0
conv2d_38_input€€€€€€€€€АА
p

 
™ "К€€€€€€€€€Ѓ
,__inference_sequential_7_layer_call_fn_29142~!"/0=>KLYZghyzГДСТJҐG
@Ґ=
3К0
conv2d_38_input€€€€€€€€€АА
p 

 
™ "К€€€€€€€€€•
,__inference_sequential_7_layer_call_fn_29393u!"/0=>KLYZghyzГДСТAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p

 
™ "К€€€€€€€€€•
,__inference_sequential_7_layer_call_fn_29434u!"/0=>KLYZghyzГДСТAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p 

 
™ "К€€€€€€€€€÷
#__inference_signature_wrapper_29193Ѓ!"/0=>KLYZghyzГДСТUҐR
Ґ 
K™H
F
conv2d_38_input3К0
conv2d_38_input€€€€€€€€€АА"=™:
8
activation_58'К$
activation_58€€€€€€€€€