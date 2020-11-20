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

untrusted(X) :- trusted(X),!, fail.
untrusted('Explanation').

modify(X,Y) :- feature(X),importance(Y),feature_importance(X, Y).

modification(X,Y,Z) :- modify(X,Y),importance(Y),sign(Z).

expected(X,'Positive','+','Increase') :-
    evaluated(X,'Positive','+','Increase'),modification(X,'Positive','+').

expected(X,'Positive','-','Decrease') :-
    evaluated(X,'Positive','-','Decrease'),modification(X,'Positive','-').

expected(X,'Negative','+','Decrease') :-
    evaluated(X,'Negative','+','Decrease'),modification(X,'Negative','+').

expected(X,'Negative','-','Increase') :-
    evaluated(X,'Negative','-','Increase'),modification(X,'Negative','-').

expected(X,'Neutral','+','Stable') :-
    evaluated(X,'Neutral','+','Stable'),modification(X,'Neutral','+').

expected(X,'Neutral','-','Stable') :-
    evaluated(X,'Neutral','-','Stable'),modification(X,'Neutral','-').

truthful(X) :-  expected(X,'Positive','+','Increase'),expected(X,'Positive','-','Decrease').
truthful(X) :-  expected(X,'Negative','+','Decrease'),expected(X,'Negative','-','Increase').
truthful(X) :- expected(X,'Neutral','+','Stable'),expected(X,'Neutral','-','Stable').

%truthful(X) :- untruthful(X), !, fail.
not(untruthful(X)) :- truthful(X),!, display(X," is not untruthful").
not(untruthful(X)) :- display(X, " is indeed untruthful").

display( Message, Goal)  :-
write( Message),
write( Goal), nl.
feature_importance('F1', 'Positive').
feature_importance('F2', 'Positive').
feature_importance('F3', 'Negative').
feature_importance('F4', 'Positive').
feature_importance('F5', 'Positive').
feature_importance('F6', 'Positive').
feature_importance('F7', 'Negative').
feature_importance('F8', 'Negative').
feature_importance('F9', 'Positive').
feature_importance('F10', 'Positive').
feature_importance('F11', 'Negative').
feature_importance('F12', 'Positive').
feature_importance('F13', 'Negative').
feature_importance('F14', 'Positive').
evaluated('F1','Positive','+','Increase').
evaluated('F2','Positive','+','Increase').
evaluated('F2','Positive','-','Decrease').
evaluated('F3','Negative','+','Decrease').
evaluated('F3','Negative','-','Increase').
evaluated('F4','Positive','+','Increase').
evaluated('F5','Positive','+','Increase').
evaluated('F5','Positive','-','Decrease').
evaluated('F6','Positive','+','Increase').
evaluated('F6','Positive','-','Decrease').
evaluated('F7','Negative','+','Decrease').
evaluated('F7','Negative','-','Increase').
evaluated('F9','Positive','+','Increase').
evaluated('F9','Positive','-','Decrease').
evaluated('F10','Positive','+','Increase').
evaluated('F10','Positive','-','Decrease').
evaluated('F11','Negative','+','Decrease').
evaluated('F11','Negative','-','Increase').
evaluated('F12','Positive','+','Increase').
evaluated('F12','Positive','-','Decrease').
evaluated('F13','Negative','+','Decrease').
evaluated('F13','Negative','-','Increase').
evaluated('F14','Positive','+','Increase').
evaluated('F14','Positive','-','Decrease').
trusted('Explanation') :- not(untruthful('F1')), writeln('F1 is untruthful'), not(untruthful('F2')), writeln('F2 is untruthful'), not(untruthful('F3')), writeln('F3 is untruthful'), not(untruthful('F4')), writeln('F4 is untruthful'), not(untruthful('F5')), writeln('F5 is untruthful'), not(untruthful('F6')), writeln('F6 is untruthful'), not(untruthful('F7')), writeln('F7 is untruthful'), not(untruthful('F8')), writeln('F8 is untruthful'), not(untruthful('F9')), writeln('F9 is untruthful'), not(untruthful('F10')), writeln('F10 is untruthful'), not(untruthful('F11')), writeln('F11 is untruthful'), not(untruthful('F12')), writeln('F12 is untruthful'), not(untruthful('F13')), writeln('F13 is untruthful'), not(untruthful('F14')), writeln('F14 is untruthful').
