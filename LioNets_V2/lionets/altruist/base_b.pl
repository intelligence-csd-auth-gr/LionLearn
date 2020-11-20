% The possible importances can be Positive, Negative or Neutral
importance('Positive').
importance('Negative').
importance('Neutral').

alteration('Increase').
alteration('Decrease').
alteration('Stable').

sign('+').
sign('-').

feature(X):-feature_importance(X, _).

untruthful(X) :- feature(X).

untrusted(X) :- trusted(X), writeln('trusted("Explanation") is valid'),!, fail.
untrusted('Explanation').

modify(X,Y) :- feature(X),importance(Y),feature_importance(X, Y).

modification(X,Y,Z) :- modify(X,Y),sign(Z).

expected(X,'Positive','+','Increase') :-
    evaluated(X,'Positive','+','Increase'),
    display(X,"'s value got higher and evaluated and the probability increased"),
    modification(X,'Positive','+'),
    display(X," has positive influence, and thus by raising its value it should increased").
expected(X,'Positive','+','Increase') :-
    modification(X,'Positive','+'),
    display(X," has positive influence, and thus by raising its value it should increased").

expected(X,'Positive','-','Decrease') :-
    evaluated(X,'Positive','-','Decrease'),
    display(X,"'s value got lower and evaluated and the probability decreased"),
    modification(X,'Positive','-'),
    display(X," has positive influence, and thus by lowering its value it should decrease").
expected(X,'Positive','-','Decrease') :-
    modification(X,'Positive','-'),
    display(X," has positive influence, and thus by lowering its value it should decrease").

expected(X,'Negative','+','Decrease') :-
    evaluated(X,'Negative','+','Decrease'),
    display(X,"'s value got higher and evaluated and the probability decreased"),
    modification(X,'Negative','+'),
    display(X," has negative influence, and thus by raising its value it should decrease").
expected(X,'Negative','+','Decrease') :-
    modification(X,'Negative','+'),
    display(X," has negative influence, and thus by raising its value it should decrease").

expected(X,'Negative','-','Increase') :-
    evaluated(X,'Negative','-','Increase'),
    display(X,"'s value got lower and evaluated and the probability increased"),
    modification(X,'Negative','-'),
    display(X," has negative influence, and thus by lowering its value it should increase").
expected(X,'Negative','-','Increase') :-
    modification(X,'Negative','-'),
    display(X," has negative influence, and thus by lowering its value it should increase").

expected(X,'Neutral','+','Stable') :-
    evaluated(X,'Neutral','+','Stable'),
    display(X,"'s value got higher and evaluated and the probability remained stable"),
    modification(X,'Neutral','+'),
    display(X," has neutral influence, and thus by raising its value it should remain stable").
expected(X,'Neutral','+','Stable') :-
    modification(X,'Neutral','+'),
    display(X," has neutral influence, and thus by raising its value it should remain stable").

expected(X,'Neutral','-','Stable') :-
    evaluated(X,'Neutral','-','Stable'),
    display(X,"'s value got lower and evaluated and the probability remained stable"),
    modification(X,'Neutral','-'),
    display(X," has neutral influence, and thus by lowering its value it should remain stable").
expected(X,'Neutral','-','Stable') :-
    modification(X,'Neutral','-'),
    display(X," has neutral influence, and thus by lowering its value it should remain stable").

truthful(X) :-  expected(X,'Positive','+','Increase'),expected(X,'Positive','-','Decrease'), display(X,' is truthful because it has positive influence and when its value raises locally we observe the probability to increase, while when its value reduces locally we observe the probability to decrease').
truthful(X) :-  expected(X,'Negative','+','Decrease'),expected(X,'Negative','-','Increase'), display(X,' is truthful because it has negative influence and when its value raises locally we observe the probability to decrease, while when its value reduces locally we observe the probability to increase').
truthful(X) :- expected(X,'Neutral','+','Stable'),expected(X,'Neutral','-','Stable'), display(X,' is truthful because it has neutral influence and when its value raises locally we observe the probability to remain stable, while when its value reduces locally we observe the probability to remain stable').


why_untruthful(X) :- expected(X,'Positive','+','Increase'), display(X,' is truthful because it has positive influence and when its value raises locally we observe the probability to increase, while when its value reduces locally we observe the probability to decrease').
why_untruthful(X) :- expected(X,'Positive','-','Decrease'), display(X,' is truthful because it has positive influence and when its value raises locally we observe the probability to increase, while when its value reduces locally we observe the probability to decrease').

why_untruthful(X) :- expected(X,'Negative','+','Decrease'), display(X,' is truthful because it has negative influence and when its value raises locally we observe the probability to decrease, while when its value reduces locally we observe the probability to increase').
why_untruthful(X) :- expected(X,'Negative','-','Increase'), display(X,' is truthful because it has negative influence and when its value raises locally we observe the probability to decrease, while when its value reduces locally we observe the probability to increase').

why_untruthful(X) :- expected(X,'Neutral','+','Stable'), display(X,' is truthful because it has neutral influence and when its value raises locally we observe the probability to remain stable, while when its value reduces locally we observe the probability to remain stable').

why_untruthful(X) :- expected(X,'Neutral','-','Stable'), display(X,' is truthful because it has neutral influence and when its value raises locally we observe the probability to remain stable, while when its value reduces locally we observe the probability to remain stable').



%truthful(X) :- untruthful(X), !, fail.
not(untruthful(X)) :- truthful(X),!.
not(untruthful(X)) :- why_untruthful(X),!.

display( Message, Goal)  :-
write( Message),
write( Goal), nl.
